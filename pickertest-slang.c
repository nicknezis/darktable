#include "common/darktable.h"
#include "common/color_picker.h"
#include "common/slang_gpu.h"
#include "common/iop_profile.h"
#include "develop/format.h"
#include "develop/imageop.h"
#include "develop/develop.h"
#include <stdio.h>

static float frand(void) { return (float)rand() / (float)RAND_MAX; }

static void report(const char *name, lib_colorpicker_stats a, lib_colorpicker_stats b, int ok)
{
  double md = 0.0;
  for(int s = 0; s < 3; s++)
    for(int c = 0; c < 4; c++)
    {
      const float x = a[s][c], y = b[s][c];
      if(x == y) continue;
      const double d = fabs((double)x - y) / fmax(1.0, fabs((double)x));
      if(d > md) md = d;
    }
  printf("%-34s slang %s  maxreldiff %.3g\n", name, ok ? "ok  " : "FAIL", md);
  if(md > 1e-4)
    for(int s = 0; s < 3; s++)
      printf("   %d cpu %g %g %g %g | slang %g %g %g %g\n", s, a[s][0], a[s][1], a[s][2], a[s][3],
             b[s][0], b[s][1], b[s][2], b[s][3]);
}

int main(int argc, char **argv)
{
  char datadir[1024], moduledir[1024];
  const char *prefix = getenv("DT_PREFIX") ? getenv("DT_PREFIX") : "/home/user/dt-install";
  snprintf(datadir, sizeof(datadir), "%s/share/darktable", prefix);
  snprintf(moduledir, sizeof(moduledir), "%s/lib/darktable", prefix);
  char *dargv[] = { "pickertest", "--library", ":memory:", "--configdir", "/tmp/pt-conf",
                    "--conf", "slang_gpu=TRUE", "--conf", "opencl=FALSE", "--datadir", datadir, "--moduledir", moduledir, NULL };
  if(dt_init(13, dargv, FALSE, FALSE, NULL)) return 1;
  dt_slang_gpu_prepare();
  printf("slang enabled %d\n", dt_slang_gpu_enabled());
  const int W = 613, H = 401;
  srand(1);
  float *rgb = dt_alloc_align_float(4 * W * H);
  float *lab = dt_alloc_align_float(4 * W * H);
  float *raw = dt_alloc_align_float(W * H);
  for(int k = 0; k < 4 * W * H; k++) rgb[k] = frand() * 1.2f - 0.05f;
  for(int k = 0; k < W * H; k++)
  {
    lab[4 * k] = frand() * 100.f;
    lab[4 * k + 1] = frand() * 160.f - 80.f;
    lab[4 * k + 2] = frand() * 160.f - 80.f;
    lab[4 * k + 3] = frand();
    raw[k] = frand();
  }
  dt_slang_gpu_image_t *irgb = dt_slang_gpu_image_new(W, H, 4, rgb);
  dt_slang_gpu_image_t *ilab = dt_slang_gpu_image_new(W, H, 4, lab);
  dt_slang_gpu_image_t *iraw = dt_slang_gpu_image_new(W, H, 1, raw);
  static dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  const dt_iop_order_iccprofile_info_t *prof = dt_ioppr_add_profile_info_to_list(&dev, DT_COLORSPACE_LIN_REC2020, "", INTENT_PERCEPTUAL);
  const dt_iop_order_iccprofile_info_t *srgb = dt_ioppr_add_profile_info_to_list(&dev, DT_COLORSPACE_SRGB, "", INTENT_PERCEPTUAL);
  printf("profiles %p %p\n", (void*)prof, (void*)srgb);

  dt_iop_roi_t roi = { .x = 3, .y = 5, .width = W, .height = H, .scale = 1.0f };
  const int boxes[3][4] = { { 0, 0, W, H }, { 100, 50, 351, 230 }, { 7, 9, 8, 10 } };
  dt_iop_buffer_dsc_t d4 = { .channels = 4, .datatype = TYPE_FLOAT };
  dt_iop_buffer_dsc_t d1 = { .channels = 1, .datatype = TYPE_FLOAT, .filters = 0x94949494 };
  dt_iop_buffer_dsc_t dx = { .channels = 1, .datatype = TYPE_FLOAT, .filters = 9u };
  const uint8_t xt[6][6] = { {1,1,0,1,1,2},{1,1,2,1,1,0},{2,0,1,0,2,1},{1,1,2,1,1,0},{1,1,0,1,1,2},{0,2,1,2,0,1} };
  memcpy(dx.xtrans, xt, sizeof(xt));

  struct { const char *n; dt_iop_buffer_dsc_t *d; float *h; dt_slang_gpu_image_t *g; int ic, pc; const dt_iop_order_iccprofile_info_t *p; } cases[] = {
    { "rgb plain", &d4, rgb, irgb, IOP_CS_RGB, IOP_CS_RGB, NULL },
    { "rgb none", &d4, rgb, irgb, IOP_CS_RGB, IOP_CS_NONE, NULL },
    { "lab plain", &d4, lab, ilab, IOP_CS_LAB, IOP_CS_LAB, NULL },
    { "lab lch", &d4, lab, ilab, IOP_CS_LAB, IOP_CS_LCH, NULL },
    { "rgb hsl", &d4, rgb, irgb, IOP_CS_RGB, IOP_CS_HSL, NULL },
    { "rgb jzczhz rec2020", &d4, rgb, irgb, IOP_CS_RGB, IOP_CS_JZCZHZ, prof },
    { "rgb jzczhz srgb(nonlinear)", &d4, rgb, irgb, IOP_CS_RGB, IOP_CS_JZCZHZ, srgb },
    { "rgb jzczhz noprofile", &d4, rgb, irgb, IOP_CS_RGB, IOP_CS_JZCZHZ, NULL },
    { "bayer", &d1, raw, iraw, IOP_CS_RAW, IOP_CS_RAW, NULL },
    { "xtrans", &dx, raw, iraw, IOP_CS_RAW, IOP_CS_RAW, NULL },
  };
  for(size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
    for(int b = 0; b < 3; b++)
      for(int dn = 0; dn < 2; dn++)
      {
        if(dn && cases[c].d->channels == 1) continue;
        lib_colorpicker_stats a, s;
        memset(s, 0, sizeof(s));
        dt_color_picker_helper(cases[c].d, cases[c].h, &roi, boxes[b], dn, a, cases[c].ic, cases[c].pc, cases[c].p);
        const int ok = dt_color_picker_helper_slang(cases[c].d, cases[c].g, &roi, boxes[b], dn, s, cases[c].ic, cases[c].pc, cases[c].p);
        char name[100];
        snprintf(name, sizeof(name), "%s box%d%s", cases[c].n, b, dn ? " denoise" : "");
        report(name, a, s, ok);
      }
  dt_slang_gpu_image_free(irgb);
  dt_slang_gpu_image_free(ilab);
  dt_slang_gpu_image_free(iraw);
  return 0;
}
