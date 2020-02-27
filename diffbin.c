#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <math.h>

int compareOutput(char *student, char *target, float epsilon);

int main(int argc, char **argv)
{
  float eps[7] = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
  int i, check;

  for(i=0;i<7;i++) {
    check = compareOutput("output.bin", "target.bin", eps[i]);
    if(check == 0) {
      printf("Output validates when testing with epsilon: %.7f \n",eps[i] );
      break;
    } else {
      printf("Output does not validate with epsilon %.7f \n", eps[i]);
    }
  }
}

int compareOutput(char *student, char *target, float epsilon)
{
    FILE *f1, *f2;
    int imax, jmax, i, j;

    float *u1, *u2, *v1, *v2, *p1, *p2;
    char *flags1, *flags2;

    if ((f1 = fopen(student, "rb"))  == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", student,
            strerror(errno));
        return 1;
    }
    if ((f2 = fopen(target, "rb"))  == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", target,
            strerror(errno));
        return 1;
    }

    fread(&imax, sizeof(int), 1, f1);
    fread(&jmax, sizeof(int), 1, f1);
    fread(&i, sizeof(int), 1, f2);
    fread(&j, sizeof(int), 1, f2);
    if (i != imax || j != jmax) {
        printf("Number of cells differ! (%dx%d vs %dx%d)\n", imax, jmax, i, j);
        return 1;
    }

    float xlength1, ylength1, xlength2, ylength2;
    fread(&xlength1, sizeof(float), 1, f1);
    fread(&ylength1, sizeof(float), 1, f1);
    fread(&xlength2, sizeof(float), 1, f2);
    fread(&ylength2, sizeof(float), 1, f2);
    if (xlength1 != xlength2 || ylength1 != ylength2) {
        printf("Image domain dimensions differ! (%gx%g vs %gx%g)\n",
            xlength1, ylength1, xlength2, ylength2);
        return 1;
    }

    u1 = malloc(sizeof(float) * (jmax + 2));
    u2 = malloc(sizeof(float) * (jmax + 2));
    v1 = malloc(sizeof(float) * (jmax + 2));
    v2 = malloc(sizeof(float) * (jmax + 2));
    p1 = malloc(sizeof(float) * (jmax + 2));
    p2 = malloc(sizeof(float) * (jmax + 2));
    flags1 = malloc(jmax + 2);
    flags2 = malloc(jmax + 2);
    if (!u1 || !u2 || !v1 || !v2 || !p1 || !p2 || !flags1 || !flags2) {
        fprintf(stderr, "Couldn't allocate enough memory.\n");
        return 1;
    }

    int diff_found = 0;
    for (i = 0; i < imax + 2 && !diff_found; i++) {
        fread(u1, sizeof(float), jmax + 2, f1);
        fread(v1, sizeof(float), jmax + 2, f1);
        fread(p1, sizeof(float), jmax + 2, f1);
        fread(flags1, 1, jmax + 2, f1);
        fread(u2, sizeof(float), jmax + 2, f2);
        fread(v2, sizeof(float), jmax + 2, f2);
        fread(p2, sizeof(float), jmax + 2, f2);
        fread(flags2, 1, jmax + 2, f2);
        for (j = 0; j < jmax + 2 && !diff_found; j++) {
            float du, dv, dp;
            int dflags;
            du = u1[j] - u2[j];
            dv = v1[j] - v2[j];
            dp = p1[j] - p2[j];
            dflags = flags1[j] - flags2[j];
            if(fpclassify(du) == FP_NAN ||
                fpclassify(dv) == FP_NAN ||
                fpclassify(dp) == FP_NAN ||
                fpclassify(du) == FP_INFINITE ||
                fpclassify(dv) == FP_INFINITE ||
                fpclassify(dp) == FP_INFINITE) {
                diff_found = 1;
                break;
            }

            if (fabs(du) > epsilon || fabs(dv) > epsilon ||
                fabs(dp) > epsilon || fabs(dflags) > epsilon) {
                diff_found = 1;
                break;
            }
        }
    }
    if (diff_found) {
        return 1;
    } else {
        return 0;
    }

}
