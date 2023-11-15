#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#define A(i, j) a[lda * i + j]
#define B(i, j) b[ldb * i + j]
#define C(i, j) c[ldc * i + j]

// apple m1 l1 cache size is 128 + 64 = 192 KB / 192 + 128 = 320 KB
// apple m1 l2 cache size is 24 MB
// cache line size is 128 bytes
// pagetable size is 16 KB

/*
About GEMM_K or kc:
1. mc = kc, since we have to maxmize (2 * mc * kc / (2 + mc + kc))
2. The equation exists provided kc << n
3. mc * kc <= K

About GEMM_M or mc:
1. The larger mc * kc, the better calculation efficiency
2. We prepare to load A into L2 cache. Avoid TLB miss (which would
stall CPU), subset of A should remains so until no longer needed.

About KERNEL_4X4, mr = 4 and nr = 4
In order to move data efficiently to registers.
Here we use C_block = A_panel x Transpose(B_panel)

nr * kc <= l1 cache / 2 = 96 kb
kc <= pagesize / 2 * 2 = 16 kb

nr = 4, kc = 8 kb
nr = 8, kc = 4 kb

registers 128 bits (32x4) v0-v31

half registers are used to store mr x nr submatrix of C (8 x 8 ?)
*/

#define GEMM_M (512)   // (384)
#define GEMM_N (4096)   // (4096)
#define GEMM_K (512)   // (256)
#define GEMM_UNROLL_4 (4)

#define min(a, b) ((a) < (b) ? (a) : (b))

static double gtod_ref_time_sec = 0.0;

extern void KERNEL_4X4(float *a, float *b, float *c, int ldc_offset, int k);
extern void KERNEL_8X8(float *a, float *b, float *c, int ldc_offset, int k);


void PackMatrixA_4x4(int m, int k, float *a_from, int lda, float *a_to);
void PackMatrixB_4x4(int k, int n, float *b_from, int ldb, float *b_to);

void PackMatrixA_8x8(int m, int k, float *a_from, int lda, float *a_to);
void PackMatrixB_8x8(int k, int n, float *b_from, int ldb, float *b_to);


void kernel_4x4_outer(int m, int n, int k, float *sa, float *sb, float *sc, int ldc) 
{
    assert(m > 0 && n > 0 && k > 0);
    assert(m % 4 == 0 && n % 4 == 0 && k % 4 == 0);

    float *a = sa, *b = sb, *c = sc;
    int ldc_offset = ldc * sizeof(float);
    int k_outer = k >> 2;

    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            // printf("%d %d\n", i, j);
            asm volatile(
                ".macro INIT4X4                     \n"
                "   fmov s16, wzr                   \n"
                "   fmov s17, s16                   \n"
                "   fmov s20, s17                   \n"
                "   fmov s21, s20                   \n"
                "   fmov s24, s21                   \n"
                "   fmov s25, s24                   \n"
                "   fmov s28, s25                   \n"
                "   fmov s29, s28                   \n"
                "   fmov s10, s29                   \n"
                "   fmov s11, s10                   \n"
                "   fmov s12, s11                   \n"
                "   fmov s13, s12                   \n"
                ".endm                              \n"
                "                                    \n"
                "mov x0, %0\n\t"
                "mov x1, %1\n\t"
                "mov x2, %2\n\t"
                "mov x3, %3\n\t"
                "mov x4, %4\n\t"
                "INIT4X4                           \n"
                // "asr x4, x4, 2\n\t"
                :
                : "r"(a), "r"(b), "r"(c), "r"(ldc_offset), "r"(k_outer)
                // : "0"(a), "1"(b), "2"(c), "3"(ldc_offset), "4"(k)
                : "x0", "x1", "x2", "x3", "x4"
            );
            KERNEL_4X4(a, b, c, ldc_offset, k_outer);
            c += 4;
            b += 4 * k;
            // a -= 4 * k;
        } // endj
        sc += ldc * 4;
        c = sc;
        a += 4 * k;
        b = sb;
    } // endi
}

void vec_mul_cpp(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int l = 0; l < k; ++l)
            {
                C(i, j) = C(i, j) * 1.0 + A(i, l) * B(l, j);
            }
        }
    }
}

void mat_mul_kernel_4x4(int m, int n, int k, float *a, int lda,
             float *b, int ldb, float *c, int ldc)
{
    // pack 之后的矩阵，实际上只会使用一小部份
    float *sa, *sb;
    sa = (float *)malloc(m * k * sizeof(float));
    sb = (float *)malloc(k * n * sizeof(float));

    int min_m, min_mm, min_n, min_k;
    // 最外层按照GEMM_M X GEMM_K, GEMM_K X GEMM_N 来切分
    for (int ms = 0; ms < m; ms += GEMM_M) { // ms表示矩阵左上角的横坐标
        min_m = min(m - ms, GEMM_M); // 为了处理边界情况，min_m最大是GEMM_M
        for (int ks = 0; ks < k; ks += min_k) { // 需要保证min_k是GEMM_UNROLL(4)的整数倍 
            min_k = k - ks;
            if (min_k >= (GEMM_K << 1)) {
                min_k = GEMM_K;
            }
            else if (min_k > GEMM_K) {
                min_k = (min_k / 2 + GEMM_UNROLL_4 - 1) & ~(GEMM_UNROLL_4 - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2) { // 保证min_n是GEMM_UNROLL(4)的整数倍
                min_n = GEMM_N;
            } else if (n > GEMM_N) {
                min_n = (n / 2 + GEMM_UNROLL_4 - 1) & ~(GEMM_UNROLL_4 - 1);
            }
            PackMatrixB_4x4(min_k, min_n, b + ks  * ldb, ldb, sb); // pack b是从上到下

            // micro kernel, split A Block to smaller Panel
            // 针对的是min_m X min_k大小的矩阵
            for (int mms = ms; mms < ms + min_m; mms += min_mm) {
                min_mm = (ms + min_m) - mms;
                if (min_mm >= 3 * GEMM_UNROLL_4) {
                    min_mm = 3 * GEMM_UNROLL_4;
                } else if (min_mm >= 2 * GEMM_UNROLL_4) {
                    min_mm = 2 * GEMM_UNROLL_4;
                } else if (min_mm > GEMM_UNROLL_4) {
                    min_mm = GEMM_UNROLL_4;
                }

                // continuous packA
                PackMatrixA_4x4(min_mm, min_k, a + mms * lda + ks, lda, sa + min_k * (mms - ms));
                kernel_4x4_outer(min_mm, min_n, min_k, sa + min_k * (mms - ms), sb, c + mms * ldc, ldc);
            }
            // the first B Block has been packed, process the last B Block
            for (int ns = min_n; ns < n; ns += min_n) {
                min_n = n - ns;
                if (min_n >= GEMM_N * 2) {
                    min_n = GEMM_N;
                } else if (min_n > GEMM_N) {
                    min_n = (min_n / 2 + GEMM_UNROLL_4 - 1) & ~(GEMM_UNROLL_4 - 1);
                }
                PackMatrixB_4x4(min_k, min_n, b + ns + ldb * ks, ldb, sb);
                kernel_4x4_outer(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }
    free(sa);
    free(sb);
}



/**
pack A means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag

Output:
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7
8 8 8 8 9 9 9 9 a a a a b b b b
c c c c d d d d e e e e f f f f

Draw it with a line
*/
void PackMatrixA_4x4(int m, int k, float *a_from, int lda, float *a_to)
{
    assert(m != 0 && k != 0 && m % 4 == 0 && k % 4 == 0);
    float *a_to_offset; // 4x4的矩阵为一个连续的一行
    
    float *a_from_offset;
    float *a_from_offset1, *a_from_offset2, *a_from_offset3, *a_from_offset4; // 分别表示原矩阵的4个连续的一行

    float a_to_1, a_to_2, a_to_3, a_to_4;
    float a_to_5, a_to_6, a_to_7, a_to_8;
    float a_to_9, a_to_10, a_to_11, a_to_12;
    float a_to_13, a_to_14, a_to_15, a_to_16;

    a_from_offset = a_from;
    a_to_offset = a_to;

    for (int i = 0; i < m; i += 4) {
        a_from_offset1 = a_from_offset;
        a_from_offset2 = a_from_offset1 + lda;
        a_from_offset3 = a_from_offset2 + lda;
        a_from_offset4 = a_from_offset3 + lda;

        a_from_offset += 4 * lda;
        for (int j = 0; j < k; j += 4) {
            a_to_1 = *a_from_offset1;
            a_to_2 = *(a_from_offset1 + 1);
            a_to_3 = *(a_from_offset1 + 2);
            a_to_4 = *(a_from_offset1 + 3);

            a_to_5 = *a_from_offset2;
            a_to_6 = *(a_from_offset2 + 1);
            a_to_7 = *(a_from_offset2 + 2);
            a_to_8 = *(a_from_offset2 + 3);

            a_to_9 = *a_from_offset3;
            a_to_10 = *(a_from_offset3 + 1);
            a_to_11 = *(a_from_offset3 + 2);
            a_to_12 = *(a_from_offset3 + 3);

            a_to_13 = *a_from_offset4;
            a_to_14 = *(a_from_offset4 + 1);
            a_to_15 = *(a_from_offset4 + 2);
            a_to_16 = *(a_from_offset4 + 3);

            *(a_to_offset) = a_to_1;
            *(a_to_offset + 1) = a_to_5;
            *(a_to_offset + 2) = a_to_9;
            *(a_to_offset + 3) = a_to_13;

            *(a_to_offset + 4) = a_to_2;
            *(a_to_offset + 5) = a_to_6;
            *(a_to_offset + 6) = a_to_10;
            *(a_to_offset + 7) = a_to_14;

            *(a_to_offset + 8) = a_to_3;
            *(a_to_offset + 9) = a_to_7;
            *(a_to_offset + 10) = a_to_11;
            *(a_to_offset + 11) = a_to_15;

            *(a_to_offset + 12) = a_to_4;
            *(a_to_offset + 13) = a_to_8;
            *(a_to_offset + 14) = a_to_12;
            *(a_to_offset + 15) = a_to_16;

            a_to_offset += 16;
            a_from_offset1 += 4;
            a_from_offset2 += 4;
            a_from_offset3 += 4;
            a_from_offset4 += 4;
        }
    }
}


/*
suppose that k and n is mutiple of 4
pack B means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag, not like pack A

Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
void PackMatrixB_4x4(int k, int n, float *b_from, int ldb, float *b_to)
{
    assert(k != 0 && n != 0 && k % 4 == 0 && n % 4 == 0);
    float * b_to_offset;

    float * b_from_offset;
    float * b_from_offset1, *b_from_offset2, *b_from_offset3, *b_from_offset4;

    float b_to_1, b_to_2, b_to_3, b_to_4;
    float b_to_5, b_to_6, b_to_7, b_to_8;
    float b_to_9, b_to_10, b_to_11, b_to_12;
    float b_to_13, b_to_14, b_to_15, b_to_16;

    b_from_offset = b_from;
    b_to_offset = b_to;

    for (int j = 0; j < n / 4; j += 1) {

        b_from_offset1 = b_from_offset;
        b_from_offset2 = b_from_offset1 + ldb;
        b_from_offset3 = b_from_offset2 + ldb;
        b_from_offset4 = b_from_offset3 + ldb;

        for (int i = 0; i < k / 4; i += 1) {

            b_to_1 = *b_from_offset1;
            b_to_2 = *(b_from_offset1 + 1);
            b_to_3 = *(b_from_offset1 + 2);
            b_to_4 = *(b_from_offset1 + 3);

            b_to_5 = *b_from_offset2;
            b_to_6 = *(b_from_offset2 + 1);
            b_to_7 = *(b_from_offset2 + 2);
            b_to_8 = *(b_from_offset2 + 3);

            b_to_9 = *b_from_offset3;
            b_to_10 = *(b_from_offset3 + 1);
            b_to_11 = *(b_from_offset3 + 2);
            b_to_12 = *(b_from_offset3 + 3);

            b_to_13 = *b_from_offset4;
            b_to_14 = *(b_from_offset4 + 1);
            b_to_15 = *(b_from_offset4 + 2);
            b_to_16 = *(b_from_offset4 + 3);

            b_from_offset1 += 4 * ldb;
            b_from_offset2 += 4 * ldb;
            b_from_offset3 += 4 * ldb;
            b_from_offset4 += 4 * ldb;

            *(b_to_offset + 0) = b_to_1;
            *(b_to_offset + 1) = b_to_2;
            *(b_to_offset + 2) = b_to_3;
            *(b_to_offset + 3) = b_to_4;

            *(b_to_offset + 4) = b_to_5;
            *(b_to_offset + 5) = b_to_6;
            *(b_to_offset + 6) = b_to_7;
            *(b_to_offset + 7) = b_to_8;

            *(b_to_offset + 8) = b_to_9;
            *(b_to_offset + 9) = b_to_10;
            *(b_to_offset + 10) = b_to_11;
            *(b_to_offset + 11) = b_to_12;

            *(b_to_offset + 12) = b_to_13;
            *(b_to_offset + 13) = b_to_14;
            *(b_to_offset + 14) = b_to_15;
            *(b_to_offset + 15) = b_to_16;

            b_to_offset += 16;

        }
        b_from_offset += 4;
    }
}

// void test_whole() {
//     int M = 1024;
//     int N = 1024;
//     int K = 1024;

//     int lda = 1024;
//     int ldb = 1024;
//     int ldc = 1024;

//     float *a = (float *)malloc(M * K * sizeof(float));
//     float *b = (float *)malloc(K * N * sizeof(float));
//     float *c = (float *)malloc(M * N * sizeof(float));
//     random_matrix(M, K, a, lda);
//     random_matrix(K, N, b, ldb);
//     memset(c, 0, M * N * sizeof(float));
//     float *c_std = (float *)malloc(M * N * sizeof(float));
//     memset(c_std, 0, M * N * sizeof(float));

//     int m = M;
//     int n = N;
//     int k = K;

//     MY_MMult(m, n, k, a, lda, b, ldb, c_std, ldc);

//     mat_mul(m, n, k, a, lda, b, ldb, c, ldc);

//     compare(a, b, m * n);
// }

// void test_perf()
// {
//     float a[16];
//     float a_packed[16];
//     float b[16];
//     float c[16];
//     float c_std[16];

//     for (int i = 0; i < 16; ++i)
//     {
//         a[i] = (float)i / 2;
//         b[i] = (float)i / 2 + 1.0f;
//     }

//     pack_a(a, 4, a_packed);

//     double start = dclock();
//     for (int i = 0; i < 1000000; ++i)
//     {
//         vec_mul_cpp(4, 4, 4, a_packed, 4, b, 4, c_std, 4);
//         memset(c_std, 0, 16 * sizeof(float));
//     }
//     double end = dclock();
//     printf("std runtime = %f\n", end - start);

//     start = dclock();
//     for (int i = 0; i < 1000000; ++i)
//     {
//         KERNEL_4X4(a, b, c);
//         memset(c, 0, 16 * sizeof(float));
//     }
//     end = dclock();
//     compare(c, c_std, 16);
//     printf("asm runtime = %f\n", end - start);

//     start = dclock();
//     for (int i = 0; i < 1000000; ++i)
//     {
//         KERNEL_4X4_V2(a, b, c);
//         memset(c, 0, 16 * sizeof(float));
//     }
//     end = dclock();
//     compare(c, c_std, 16);
//     printf("asm_v2 runtime = %f\n", end - start);

//     printf("Hello World\n");
// }

// void test_pack()
// {
//     float *a = (float *)malloc(16 * 4 * sizeof(float));
//     float *a_pack_ori = (float *)malloc(16 * 4 * sizeof(float));
//     float *a_pack_new = (float *)malloc(16 * 4 * sizeof(float));

//     int m, k, lda;
//     m = k = lda = 8;

//     random_matrix(m, k, a, lda);

//     PackMatrixB(m, k, a, lda, a_pack_new);
//     packB_4(m, k, a, lda, a_pack_ori);
//     compare(a_pack_ori, a_pack_new, 64);

//     free(a);
//     free(a_pack_ori);
//     free(a_pack_new);
// }

// int main()
// {
//     test_whole();
//     printf("Hello World\n");
//     return 0;
// }
