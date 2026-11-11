#include <glib/gstdio.h>
// CPU vs OpenCL vs double-accumulated reference for the color picker statistics
#include "common/darktable.h"
#include "common/color_picker.h"
#include "common/opencl.h"
#include "common/iop_profile.h"
#include "common/colorspaces_inline_conversions.h"
#include "develop/format.h"
#include "develop/imageop.h"
#include "develop/develop.h"
#include <stdio.h>

static float frand(void) { return (float)rand() / (float)RAND_MAX; }

// double-accumulated reference, same per-pixel float conversion as the CPU
static void reference(const float *px, int W, const int *box, int mode, const dt_iop_order_iccprofile_info_t *p,
                      double mean[4])
{
  double acc[4] = { 0 };
  for(int y = box[1]; y < box[3]; y++)
    for(int x = box[0]; x < box[2]; x++)
    {
      const float *in = px + 4 * ((size_t)y * W + x);
      dt_aligned_pixel_t pick = { in[0], in[1], in[2], in[3] };
      if(mode == 1) dt_Lab_2_LCH(in, pick);
      else if(mode == 2) dt_RGB_2_HSL(in, pick);
      else if(mode == 3)
      {
        dt_aligned_pixel_t D50, D65, Jab;
        dt_ioppr_rgb_matrix_to_xyz(in, D50, p->matrix_in_transposed, p->lut_in, p->unbounded_coeffs_in, p->lutsize, p->nonlinearlut);
        dt_XYZ_D50_2_XYZ_D65(D50, D65);
        dt_XYZ_2_JzAzBz(D65, Jab);
        dt_JzAzBz_2_JzCzhz(Jab, pick);
      }
      if(mode) pick[3] = pick[2] < 0.5f ? pick[2] + 0.5f : pick[2] - 0.5f;
      for(int c = 0; c < 4; c++) acc[c] += pick[c];
    }
  const double n = (double)(box[2] - box[0]) * (box[3] - box[1]);
  for(int c = 0; c < 4; c++) mean[c] = acc[c] / n;
}

static double dmax(const float *a, const double *b, int k)
{
  double m = 0;
  for(int c = 0; c < k; c++) m = fmax(m, fabs((double)a[c] - b[c]));
  return m;
}

int main(int argc, char **argv)
{
  char datadir[1024], moduledir[1024];
  const char *prefix = getenv("DT_PREFIX") ? getenv("DT_PREFIX") : "/home/user/dt-install";
  snprintf(datadir, sizeof(datadir), "%s/share/darktable", prefix);
  snprintf(moduledir, sizeof(moduledir), "%s/lib/darktable", prefix);
  char *dargv[] = { "clpicktest", "--library", ":memory:", "--configdir", "/tmp/cpt-conf",
                    "--conf", "opencl=TRUE", "--conf", "clplatform_portablecomputinglanguage=TRUE", "--conf", "clplatform_other=TRUE", "--datadir", datadir,
                    "--moduledir", moduledir, "-d", "opencl", NULL };
  if(dt_init(17, dargv, FALSE, FALSE, NULL)) return 1;
  if(!darktable.opencl->inited || darktable.opencl->num_devs < 1) { printf("no opencl inited %d num %d\n", darktable.opencl ? darktable.opencl->inited : -1, darktable.opencl ? darktable.opencl->num_devs : -1); return 1; }
  const int devid = 0;
  int W = 1600, H = 1000;
  FILE *pf = getenv("PFM") ? g_fopen(getenv("PFM"), "rb") : NULL;
  float *pfm = NULL;
  if(pf)
  {
    char hdr[3]; float sc;
    if(fscanf(pf, "%2s %d %d %f", hdr, &W, &H, &sc) != 4) return 1;
    fgetc(pf);
    pfm = malloc(sizeof(float) * 3 * W * H);
    if(fread(pfm, sizeof(float) * 3, (size_t)W * H, pf) != (size_t)W * H) return 1;
    fclose(pf);
    printf("real image %dx%d\n", W, H);
  }
  srand(1);
  float *rgb = dt_alloc_align_float(4 * W * H);
  float *lab = dt_alloc_align_float(4 * W * H);
  float *labn = dt_alloc_align_float(4 * W * H); // a smooth reddish area with hue near 0: noisy b around 0
  for(int k = 0; k < W * H; k++)
  {
    // a smooth gradient plus noise, roughly photo-like values
    const int x = k % W, y = k / W;
    rgb[4 * k + 0] = 0.18f + 0.3f * x / W + 0.02f * frand();
    rgb[4 * k + 1] = 0.10f + 0.2f * y / H + 0.02f * frand();
    rgb[4 * k + 2] = 0.05f + 0.1f * (x + y) / (W + H) + 0.02f * frand();
    rgb[4 * k + 3] = 0.0f;
    lab[4 * k] = 30.0f + 40.0f * x / W + frand();
    lab[4 * k + 1] = -20.0f + 40.0f * y / H + frand();
    lab[4 * k + 2] = 10.0f + 10.0f * x / W + frand();
    lab[4 * k + 3] = 0.0f;
    labn[4 * k] = 50.0f + frand();
    labn[4 * k + 1] = 30.0f + frand();
    labn[4 * k + 2] = 0.5f * (frand() - 0.5f);
    labn[4 * k + 3] = 0.0f;
  }
  if(pfm)
    for(int k = 0; k < W * H; k++)
    {
      const int src = (H - 1 - k / W) * W + k % W; // pfm rows are bottom-up
      dt_aligned_pixel_t c = { pfm[3 * src], pfm[3 * src + 1], pfm[3 * src + 2], 0.0f }, XYZ, L;
      for(int i = 0; i < 4; i++) rgb[4 * k + i] = c[i];
      dt_sRGB_to_XYZ(c, XYZ);
      dt_XYZ_to_Lab(XYZ, L);
      for(int i = 0; i < 3; i++) lab[4 * k + i] = labn[4 * k + i] = L[i];
      lab[4 * k + 3] = labn[4 * k + 3] = 0.0f;
    }
  static dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  const dt_iop_order_iccprofile_info_t *prof = dt_ioppr_add_profile_info_to_list(&dev, DT_COLORSPACE_LIN_REC2020, "", INTENT_PERCEPTUAL);

  cl_mem drgb = dt_opencl_copy_host_to_image(devid, rgb, W, H, 4 * sizeof(float));
  cl_mem dlab = dt_opencl_copy_host_to_image(devid, lab, W, H, 4 * sizeof(float));
  cl_mem dlabn = dt_opencl_copy_host_to_image(devid, labn, W, H, 4 * sizeof(float));

  dt_iop_roi_t roi = { .x = 0, .y = 0, .width = W, .height = H, .scale = 1.0f };
  const int boxes[3][4] = { { 0, 0, W, H }, { 100, 50, 351, 230 }, { 700, 400, 705, 405 } };
  const int boxes_real[3][4] = { { 0, 0, W, H }, { W / 4, H / 4, W / 2, H / 2 }, { W / 2, H / 2, W / 2 + 5, H / 2 + 5 } };
  dt_iop_buffer_dsc_t d4 = { .channels = 4, .datatype = TYPE_FLOAT };
  struct { const char *n; float *h; cl_mem g; int ic, pc, mode; const dt_iop_order_iccprofile_info_t *p; } cases[] = {
    { "rgb plain", rgb, drgb, IOP_CS_RGB, IOP_CS_RGB, 0, NULL },
    { "lab plain", lab, dlab, IOP_CS_LAB, IOP_CS_LAB, 0, NULL },
    { "lab lch", lab, dlab, IOP_CS_LAB, IOP_CS_LCH, 1, NULL },
    { "lab lch hue~0", labn, dlabn, IOP_CS_LAB, IOP_CS_LCH, 1, NULL },
    { "rgb hsl", rgb, drgb, IOP_CS_RGB, IOP_CS_HSL, 2, NULL },
    { "rgb jzczhz", rgb, drgb, IOP_CS_RGB, IOP_CS_JZCZHZ, 3, prof },
  };
  for(size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
  {
    lib_colorpicker_stats a;
    double best = 1e9;
    for(int r = 0; r < 15; r++)
    {
      const gint64 t0 = g_get_monotonic_time();
      dt_color_picker_helper(&d4, cases[c].h, &roi, boxes[0], 0, a, cases[c].ic, cases[c].pc, cases[c].p);
      best = fmin(best, (g_get_monotonic_time() - t0) / 1000.0);
    }
    printf("cpu time %-18s %.2f ms (1.6 MP box)\n", cases[c].n, best);
  }
  if(getenv("TIMEONLY")) return 0;
  printf("%-18s %-5s %-3s | %-10s %-10s %-10s | %-10s %-10s\n", "case", "box", "dn",
         "cl-cpu mean", "cpu-ref", "cl-ref", "cl-cpu min", "cl-cpu max");
  for(size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
    for(int b = 0; b < 3; b++)
      for(int dn = 0; dn < 2; dn++)
      {
        lib_colorpicker_stats a, s;
        dt_color_picker_helper(&d4, cases[c].h, &roi, (pfm ? boxes_real[b] : boxes[b]), dn, a, cases[c].ic, cases[c].pc, cases[c].p);
        const int ok = dt_color_picker_helper_cl(devid, &d4, cases[c].g, &roi, (pfm ? boxes_real[b] : boxes[b]), dn, s, cases[c].ic, cases[c].pc, cases[c].p);
        double ref[4];
        reference(cases[c].h, W, (pfm ? boxes_real[b] : boxes[b]), cases[c].mode, cases[c].p, ref);
        const int k = cases[c].mode ? 4 : 3;
        double dm = 0, dmin = 0, dmax2 = 0;
        for(int i = 0; i < k; i++)
        {
          dm = fmax(dm, fabs(a[DT_PICK_MEAN][i] - s[DT_PICK_MEAN][i]));
          dmin = fmax(dmin, fabs(a[DT_PICK_MIN][i] - s[DT_PICK_MIN][i]));
          dmax2 = fmax(dmax2, fabs(a[DT_PICK_MAX][i] - s[DT_PICK_MAX][i]));
        }
        printf("%-18s box%d  %d   | %-10.3g %-10s %-10s | %-10.3g %-10.3g %s\n", cases[c].n, b, dn, dm,
               dn ? "-" : g_strdup_printf("%.3g", dmax(a[DT_PICK_MEAN], ref, k)),
               dn ? "-" : g_strdup_printf("%.3g", dmax(s[DT_PICK_MEAN], ref, k)), dmin, dmax2, ok ? "" : "CL FAIL");
        if(argc > 1)
          for(int i = 0; i < k; i++)
            printf("     ch%d mean cpu %.9g cl %.9g ref %.9g\n", i, a[DT_PICK_MEAN][i], s[DT_PICK_MEAN][i], ref[i]);
      }
  return 0;
}
