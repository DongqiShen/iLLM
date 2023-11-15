/* Routine for computing C = A * B + C */
void packB_4(int k, int n, float *from, int ldb, float *to);
void packA_4(int m, int k, float *from, int lda, float *to);
void kernel_4x4_val(int m, int n, int k, float *sa, float *sb, float *sc, int ldc);

void MY_MMult(int m, int n, int k, float *a, int lda,
              float *b, int ldb, float *c, int ldc);