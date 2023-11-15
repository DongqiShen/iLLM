#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <arm_neon.h>
#include <stddef.h>
#include "utils.h"

static double gtod_ref_time_sec = 0.0;

void compare(float *a, float *b, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (a[i] != b[i])
        {
            printf("a[%d] = %f, b[%d] = %f\n", i, a[i], i, b[i]);
        }
    }
}

float compare_matrices(int m, int n, float *a, int lda, float *b, int ldb)
{
  int i, j;
  float max_diff = 0.0, diff;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ){
      diff = abs(a[i*lda+j] - b[i*ldb+j]);
      max_diff = ( diff > max_diff ? diff : max_diff );
    }

  return max_diff;
}

void random_matrix(int m, int n, float *a, int lda)
{
    double drand48();
    int i, j;

    for (j = 0; j < n; j++)
        for (i = 0; i < m; i++)
            a[i * lda + j] = 2.0 * (float)drand48() - 1.0;
}

double dclock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double)tv.tv_sec;

    norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

void printf_array(float *array, int len, char *path)
{
    FILE *fp = fopen(path, "w");
    if (fp != NULL) {
        for (int i = 0; i < len; ++i) {
            fprintf(fp, "array[%d] = %f\n", i, array[i]);
        }
    }
}