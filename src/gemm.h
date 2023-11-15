
extern void KERNEL_4X4(float *a, float *b, float *c, int ldc_offset);

extern void KERNEL_4X4_V2(float *a, float *b, float *c);

void mat_mul_kernel_4x4(int m, int n, int k, float *a, int lda,
             float *b, int ldb, float *c, int ldc);

void kernel_4x4_outer(int m, int n, int k, float *sa, float *sb, float *sc, int ldc);

void vec_mul_cpp(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc);

void PackMatrixA_4x4(int m, int k, float *a_from, int lda, float *a_to);
void PackMatrixB_4x4(int k, int n, float *b_from, int ldb, float *b_to);
