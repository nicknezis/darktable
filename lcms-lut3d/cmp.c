// dE76 between two PFM renders, both converted to Lab through the given profile
#include <lcms2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float *readpfm(const char *f, int *w, int *h)
{
  FILE *fp = fopen(f, "rb"); char t[3]; float s;
  if(!fp || fscanf(fp, "%2s %d %d %f", t, w, h, &s) != 4) { fprintf(stderr, "bad %s\n", f); exit(1); }
  fgetc(fp);
  const int ch = t[1] == 'F' ? 3 : 1;
  float *b = malloc(sizeof(float) * 3 * *w * *h);
  if(fread(b, sizeof(float) * ch, (size_t)*w * *h, fp) != (size_t)*w * *h) { fprintf(stderr, "short %s\n", f); exit(1); }
  fclose(fp); return b;
}
int main(int argc, char **argv)
{
  int w, h, w2, h2;
  float *a = readpfm(argv[1], &w, &h), *b = readpfm(argv[2], &w2, &h2);
  if(w != w2 || h != h2) { printf("size differs\n"); return 1; }
  cmsHPROFILE p;
  if(!strcmp(argv[3], "rec2020"))
  {
    cmsCIExyY wp = { 0.3127, 0.3290, 1.0 };
    cmsCIExyYTRIPLE prim = { { 0.708, 0.292, 1 }, { 0.170, 0.797, 1 }, { 0.131, 0.046, 1 } };
    cmsToneCurve *lin = cmsBuildGamma(NULL, 1.0), *lins[3] = { lin, lin, lin };
    p = cmsCreateRGBProfile(&wp, &prim, lins);
  }
  else p = cmsOpenProfileFromFile(argv[3], "r");
  cmsHPROFILE lab = cmsCreateLab4Profile(NULL);
  cmsHTRANSFORM x = cmsCreateTransform(p, TYPE_RGB_FLT, lab, TYPE_Lab_FLT, INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOCACHE);
  const size_t n = (size_t)w * h;
  float *la = malloc(12 * n), *lb = malloc(12 * n);
  cmsDoTransform(x, a, la, n); cmsDoTransform(x, b, lb, n);
  double sum = 0, mx = 0, maxabs = 0; size_t o1 = 0, o2 = 0, cyan_a = 0, cyan_b = 0, cyan_mis = 0;
  for(size_t k = 0; k < n; k++)
  {
    cmsCIELab A = { la[3*k], la[3*k+1], la[3*k+2] }, B = { lb[3*k], lb[3*k+1], lb[3*k+2] };
    const double d = cmsDeltaE(&A, &B);
    sum += d; if(d > mx) mx = d; o1 += d > 1; o2 += d > 2;
    for(int c = 0; c < 3; c++) { double e = fabs(a[3*k+c] - b[3*k+c]); if(e > maxabs) maxabs = e; }
    const int ca = a[3*k] == 0 && a[3*k+1] == 1 && a[3*k+2] == 1, cb = b[3*k] == 0 && b[3*k+1] == 1 && b[3*k+2] == 1;
    cyan_a += ca; cyan_b += cb; cyan_mis += ca != cb;
  }
  printf("%dx%d mean dE %.4f max %.3f  >1: %.4f%%  >2: %.4f%%  max|diff| %.5f", w, h, sum / n, mx, 100.0 * o1 / n, 100.0 * o2 / n, maxabs);
  if(cyan_a || cyan_b) printf("  cyan %.2f%% vs %.2f%%, differ %.3f%%", 100.0*cyan_a/n, 100.0*cyan_b/n, 100.0*cyan_mis/n);
  printf("\n");
  return 0;
}
