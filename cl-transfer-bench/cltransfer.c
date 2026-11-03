// standalone OpenCL host<->device transfer benchmark, mirrors darktable's copy paths
// build: macOS  cc -O2 cltransfer.c -framework OpenCL -o cltransfer
//        Linux  cc -O2 cltransfer.c -lOpenCL -o cltransfer
// run:   ./cltransfer            (lists devices)
//        ./cltransfer <platform> <device> [reps]
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

#define CK(x) do { cl_int e_ = (x); if(e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d err %d\n", __FILE__, __LINE__, e_); exit(1); } } while(0)

static const char *src =
  "__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
  "__kernel void touch(__read_only image2d_t img, __global float *out, int w, int h)\n"
  "{ const int x = get_global_id(0), y = get_global_id(1);\n"
  "  if(x >= w || y >= h) return;\n"
  "  float4 p = read_imagef(img, smp, (int2)(x, y));\n"
  "  if(p.x == -12345.0f) out[0] = p.y; }\n";

static cl_context ctx;
static cl_command_queue q;
static cl_kernel ktouch;
static cl_mem dummy;
static const cl_image_format fmt = { CL_RGBA, CL_FLOAT };

static cl_mem new_image(int w, int h, cl_mem_flags flags, void *host)
{
  cl_int err;
  cl_mem m = clCreateImage2D(ctx, flags, &fmt, w, h, 0, host, &err);
  CK(err);
  return m;
}

// make sure lazily transferred data really reaches the device
static void touch(cl_mem img, int w, int h)
{
  CK(clSetKernelArg(ktouch, 0, sizeof(cl_mem), &img));
  CK(clSetKernelArg(ktouch, 1, sizeof(cl_mem), &dummy));
  CK(clSetKernelArg(ktouch, 2, sizeof(int), &w));
  CK(clSetKernelArg(ktouch, 3, sizeof(int), &h));
  const size_t gws[2] = { (w + 15) / 16 * 16, (h + 15) / 16 * 16 };
  CK(clEnqueueNDRangeKernel(q, ktouch, 2, NULL, gws, NULL, 0, NULL, NULL));
  CK(clFinish(q));
}

static int cmp(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static void report(const char *name, double *t, int reps, size_t bytes, double sub)
{
  qsort(t, reps, sizeof(double), cmp);
  const double med = t[reps / 2] - sub;
  printf("  %-44s %8.1f ms  %6.2f GB/s\n", name, 1e3 * med, bytes / med / 1e9);
}

static void bench(int w, int h, int reps)
{
  const size_t bytes = (size_t)w * h * 4 * sizeof(float);
  const size_t region[3] = { w, h, 1 }, origin[3] = { 0, 0, 0 };
  float *host = malloc(bytes);
  float *back = malloc(bytes);
  for(size_t i = 0; i < bytes / sizeof(float); i++) host[i] = (float)(i & 1023) * 0.001f;
  memset(back, 0, bytes);
  double t[64];
  cl_int err;

  printf("%dx%d RGBA float, %.1f MB, median of %d\n", w, h, bytes / 1e6, reps);

  // kernel cost on a resident image, subtracted from the "touch" variants
  cl_mem res = new_image(w, h, CL_MEM_READ_WRITE, NULL);
  CK(clEnqueueWriteImage(q, res, CL_TRUE, origin, region, 0, 0, host, 0, NULL, NULL));
  touch(res, w, h);
  for(int r = 0; r < reps; r++) { double t0 = now(); touch(res, w, h); t[r] = now() - t0; }
  qsort(t, reps, sizeof(double), cmp);
  const double ktime = t[reps / 2];
  printf("  %-44s %8.1f ms\n", "touch kernel on resident image", 1e3 * ktime);

  // A: darktable upload path, clCreateImage(COPY_HOST_PTR) then first use
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    cl_mem m = new_image(w, h, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, host);
    touch(m, w, h);
    t[r] = now() - t0;
    clReleaseMemObject(m);
  }
  report("up A: CreateImage COPY_HOST_PTR (dt path)", t, reps, bytes, ktime);

  // A0: allocation only: new image, first use by the kernel, no host data
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    cl_mem m = new_image(w, h, CL_MEM_READ_WRITE, NULL);
    touch(m, w, h);
    t[r] = now() - t0;
    clReleaseMemObject(m);
  }
  report("alloc A0: new image + first use, no copy", t, reps, bytes, ktime);

  // A2: new image each time (no host ptr), then blocking WriteImage: separates allocation from copy method
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    cl_mem m = new_image(w, h, CL_MEM_READ_WRITE, NULL);
    CK(clEnqueueWriteImage(q, m, CL_TRUE, origin, region, 0, 0, host, 0, NULL, NULL));
    touch(m, w, h);
    t[r] = now() - t0;
    clReleaseMemObject(m);
  }
  report("up A2: new image + WriteImage", t, reps, bytes, ktime);

  // B: preallocated image, blocking WriteImage from pageable host memory
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    CK(clEnqueueWriteImage(q, res, CL_TRUE, origin, region, 0, 0, host, 0, NULL, NULL));
    touch(res, w, h);
    t[r] = now() - t0;
  }
  report("up B: WriteImage pageable, image reused", t, reps, bytes, ktime);

  // C: pinned staging buffer (ALLOC_HOST_PTR), memcpy into the mapping, CopyBufferToImage
  cl_mem pin = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, NULL, &err);
  CK(err);
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    void *p = clEnqueueMapBuffer(q, pin, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes, 0, NULL, NULL, &err);
    CK(err);
    memcpy(p, host, bytes);
    CK(clEnqueueUnmapMemObject(q, pin, p, 0, NULL, NULL));
    CK(clEnqueueCopyBufferToImage(q, pin, res, 0, origin, region, 0, NULL, NULL));
    touch(res, w, h);
    t[r] = now() - t0;
  }
  report("up C: pinned staging + memcpy + CopyBufToImg", t, reps, bytes, ktime);

  // D: as C but without the memcpy (data already in pinned memory)
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    CK(clEnqueueCopyBufferToImage(q, pin, res, 0, origin, region, 0, NULL, NULL));
    touch(res, w, h);
    t[r] = now() - t0;
  }
  report("up D: pinned buffer -> image, no memcpy", t, reps, bytes, ktime);

  // E: plain device buffer, WriteBuffer from pageable (no image layout)
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
  CK(err);
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    CK(clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, host, 0, NULL, NULL));
    t[r] = now() - t0;
  }
  report("up E: WriteBuffer pageable (buffer, not image)", t, reps, bytes, 0.0);

  // F: darktable readback path, blocking ReadImage into pageable host memory
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    CK(clEnqueueReadImage(q, res, CL_TRUE, origin, region, 0, 0, back, 0, NULL, NULL));
    t[r] = now() - t0;
  }
  report("down F: ReadImage pageable (dt path)", t, reps, bytes, 0.0);

  // G: CopyImageToBuffer into pinned staging, map, memcpy out
  for(int r = 0; r < reps; r++)
  {
    double t0 = now();
    CK(clEnqueueCopyImageToBuffer(q, res, pin, origin, region, 0, 0, NULL, NULL));
    void *p = clEnqueueMapBuffer(q, pin, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err);
    CK(err);
    memcpy(back, p, bytes);
    CK(clEnqueueUnmapMemObject(q, pin, p, 0, NULL, NULL));
    CK(clFinish(q));
    t[r] = now() - t0;
  }
  report("down G: image -> pinned staging + memcpy", t, reps, bytes, 0.0);

  if(memcmp(host, back, bytes)) printf("  WARNING: readback differs from upload\n");

  clReleaseMemObject(buf);
  clReleaseMemObject(pin);
  clReleaseMemObject(res);
  free(host);
  free(back);
}

int main(int argc, char **argv)
{
  cl_platform_id plat[8];
  cl_uint np = 0;
  CK(clGetPlatformIDs(8, plat, &np));
  if(argc < 3)
  {
    for(cl_uint p = 0; p < np; p++)
    {
      cl_device_id dev[8];
      cl_uint nd = 0;
      clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_ALL, 8, dev, &nd);
      for(cl_uint d = 0; d < nd; d++)
      {
        char name[256];
        clGetDeviceInfo(dev[d], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("platform %u device %u: %s\n", p, d, name);
      }
    }
    return 0;
  }
  const int pi = atoi(argv[1]), di = atoi(argv[2]);
  const int reps = argc > 3 ? atoi(argv[3]) : 7;
  cl_device_id dev[8];
  cl_uint nd = 0;
  CK(clGetDeviceIDs(plat[pi], CL_DEVICE_TYPE_ALL, 8, dev, &nd));
  char name[256];
  clGetDeviceInfo(dev[di], CL_DEVICE_NAME, sizeof(name), name, NULL);
  printf("device: %s\n", name);
  cl_int err;
  ctx = clCreateContext(NULL, 1, &dev[di], NULL, NULL, &err);
  CK(err);
  q = clCreateCommandQueue(ctx, dev[di], 0, &err);
  CK(err);
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
  CK(err);
  CK(clBuildProgram(prog, 1, &dev[di], "", NULL, NULL));
  ktouch = clCreateKernel(prog, "touch", &err);
  CK(err);
  dummy = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16, NULL, &err);
  CK(err);

  bench(2913, 1859, reps);  // 100% view crop from the Intel Mac test
  bench(6192, 4128, reps);  // full FX30 frame, what demosaicscale used to upload
  return 0;
}
