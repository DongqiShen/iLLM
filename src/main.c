#include "common.h"
#include "gemm.h"
#include "utils.h"
#include "validation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void kernel_4x4_outer(int m, int n, int k, float *sa, float *sb, float *sc, int ldc);

void kernel_4x4_val(int m, int n, int k, float *sa, float *sb, float *sc, int ldc);

void mat_mul_kernel_4x4(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc);

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc);

void test_kernel_4x4()
{
    int m = 8, n = 8, k = 8;
    int lda = 8, ldb = 8, ldc = 8;

    float *a = (float *)malloc(m * k * sizeof(float));
    float *b = (float *)malloc(k * n * sizeof(float));
    float *c = (float *)malloc(m * n * sizeof(float));
    float *c_val = (float *)malloc(m * n * sizeof(float));

    random_matrix(m, k, a, lda);
    random_matrix(k, n, b, ldb);
    memset(c, 0, m * n * sizeof(float));
    memset(c_val, 0, m * n * sizeof(float));

    kernel_4x4_outer(m, n, k, a, b, c, ldc);
    kernel_4x4_val(m, n, k, a, b, c_val, ldc);

    compare(c, c_val, m * n);
}

void test_kernel_4x4_asm()
{
    int m = 8, n = 8, k = 8;
    int lda = 8, ldb = 8, ldc = 8;

    int ldc_offset = ldc * sizeof(ldc);

    float *a = (float *)malloc(m * k * sizeof(float));
    float *b = (float *)malloc(k * n * sizeof(float));
    float *c = (float *)malloc(m * n * sizeof(float));

    random_matrix(m, k, a, lda);
    random_matrix(k, n, b, ldb);
    memset(c, 0, m * n * sizeof(float));
    // KERNEL_4X4(a, b, c, k);
    kernel_4x4_outer(m, n, k, a, b, c, ldc);
    // KERNEL_4X4_V2(a, b, c);

    free(a);
    free(b);
    free(c);
}

void test_MMult()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    int lda = 1024;
    int ldb = 1024;
    int ldc = 1024;

    float *a = (float *)malloc(M * K * sizeof(float));
    float *b = (float *)malloc(K * N * sizeof(float));
    float *c = (float *)malloc(M * N * sizeof(float));

    float *pack_a = (float *)malloc(M * K * sizeof(float));
    float *pack_b = (float *)malloc(K * N * sizeof(float));

    random_matrix(M, K, a, lda);
    random_matrix(K, N, b, ldb);
    
    float *c_std = (float *)malloc(M * N * sizeof(float));

    memset(c, 0, M * N * sizeof(float));
    memset(c_std, 0, M * N * sizeof(float));

    int m = M;
    int n = N;
    int k = K;

    PackMatrixA_4x4(m, k, a, lda, pack_a);
    PackMatrixB_4x4(k, n, a, ldb, pack_b);

    double start = dclock();
    for (int i = 0; i < 100; ++i)
    {
        mat_mul_kernel_4x4(m, n, k, pack_a, lda, pack_b, ldb, c, ldc);
        memset(c, 0, m * n * sizeof(float));
    }
    double end = dclock();
    printf("my mmult runtime = %f\n", (end - start)/100);

    start = dclock();
    for (int i = 0; i < 100; ++i)
    {
        MY_MMult(m, n, k, pack_a, lda, pack_b, ldb, c_std, ldc);
        memset(c_std, 0, m * n * sizeof(float));
    }
    end = dclock();
    printf("std runtime = %f\n", (end - start)/100);

    compare(c_std, c, m * n);
    free(a);
    free(b);
    free(c);
    free(c_std);
}

void test_pack()
{

    int M = 4096;
    int N = 4096;
    int K = 4096;

    int lda = 4096;
    int ldb = 4096;
    int ldc = 4096;

    float *a = (float *)malloc(M * K * sizeof(float));
    float *a_pack_ori = (float *)malloc(M * K * sizeof(float));
    float *a_pack_new = (float *)malloc(M * K * sizeof(float));

    int m = M;
    int n = N;
    int k = K;


    random_matrix(m, k, a, lda);

    printf_array(a, m * k, "output/origin.txt");

    PackMatrixA_4x4(m, k, a, lda, a_pack_new);
    packA_4(m, k, a, lda, a_pack_ori);

    printf_array(a_pack_new, m * k, "output/pack_new.txt");
    printf_array(a_pack_ori, m * k, "output/pack_ori.txt");

    compare(a_pack_ori, a_pack_new, m * k);

    free(a);
    free(a_pack_ori);
    free(a_pack_new);
}

void test_perf()
{
    int DIM_BEGIN = 40;
    int DIM_END = 4096;
    int DIM_STEP = 40;
    for (int dim = DIM_BEGIN; dim <= DIM_END; dim += DIM_STEP)
    {
        int m = dim;
        int n = dim;
        int k = dim;

        int lda = dim;
        int ldb = dim;
        int ldc = dim;

        float gflops = 2.0 * m * n * k * 1.0e-09;
        
        float *a = (float *)malloc(m * k * sizeof(float));
        float *b = (float *)malloc(k * n * sizeof(float));
        float *c = (float *)malloc(m * n * sizeof(float));
        float *cold = (float *)malloc(m * n * sizeof(float));
        float *cref = (float *)malloc(m * n * sizeof(float));

        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);

        double start = dclock();
        for (int i = 0; i < 3; ++i) {
            memset(cref, 0, m * n * sizeof(float));
            vec_mul_cpp(m, n, k, a, lda, b, ldb, cref, ldc);
            // mat_mul_kernel_4x4(m, n, k, a, lda, b, ldb, cref, ldc);
        }
        double end = dclock();
        printf("std     : %d %le\n", dim, gflops / (end - start) * 3);

        start = dclock();
        for (int i = 0; i < 3; ++i) {
            memset(c, 0, m * n * sizeof(float));
            mat_mul_kernel_4x4(m, n, k, a, lda, b, ldb, c, ldc);
            // MY_MMult(m, n, k, a, lda, b, ldb, c, ldc);
        }
        end = dclock();
        float diff = compare_matrices(m, n, c, lda, cref, ldb);
        printf("My MMult: %d %le %le\n", dim, gflops / (end - start) * 3, diff);

        // compare(c, cref, m * n);
        free(a);
        free(b);
        free(c);
        free(cold);
        free(cref);
    }
}

int main()
{
    // test_kernel_4x4();
    // test_MMult();
    // test_pack();
    test_perf();
    printf("Hello World!\n");
}