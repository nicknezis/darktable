# M4 Pro test: color picker CPU vs OpenCL vs Slang/Metal, before and after the double row sums

Fork PR #158 (branch slang-color-picker-double-sums, 6d5e61d) = slang-benchmark 4d8726c + one commit
in src/common/color_picker.c (CPU picker row sums added in double).

Rules: check ~/workspace/darktable-slang.lock, wait while it exists, create it before building,
remove it when done. Never delete anything on the Mac without Nick's OK. Fetch over HTTPS from
https://github.com/nicknezis/darktable. Do not switch the branch of the main checkout: use a
git worktree inside ~/workspace/darktable-slang (e.g. wt-picker, add it to .git/info/exclude).

## Steps

1. Build base 4d8726c and PR 6d5e61d (Release, USE_SLANG=ON, OpenCL on), `cmake --build` rc=0.
   Build each into its own build dir; an existing build dir of slang-benchmark may be reused
   for base if it is at 4d8726c.
2. Test programs, in this folder (copy them to the Mac):
   - clpicktest.c: CPU helper vs `dt_color_picker_helper_cl()` (Apple OpenCL) vs a double
     reference. Env: DT_PREFIX = the build's install prefix (or set it so that
     $DT_PREFIX/share/darktable and $DT_PREFIX/lib/darktable exist; a `cmake --install` into
     a scratch prefix inside the worktree is fine), PFM = optional real image (see 3),
     TIMEONLY=1 = only CPU timing.
   - pickertest-slang.c: CPU helper vs `dt_color_picker_helper_slang()` (Metal).
   Compile EACH against the same source tree and build as the libdarktable it links
   (struct layouts differ between trees: compiling against another tree's headers gives
   garbage). Take the include/define flags from that build's compile_commands.json entry
   for src/common/color_picker.c; link -ldarktable plus gtk+-3.0, lcms2, OpenMP as needed.
   The clplatform conf keys in clpicktest.c are for PoCL; they are harmless on Apple.
3. Real image: export DSC04562.ARW (Slang_Results or project files) with its .xmp via
   darktable-cli to a 32-bit PFM at --width 2400 (`--core --conf plugins/imageio/format/pfm/bpp=32`).
4. For base and PR, run: clpicktest (synthetic), clpicktest with PFM, pickertest-slang.
   Check that OpenCL is the Apple GPU device (not CPU). Also 3 alternating TIMEONLY runs
   per build for CPU timing.
5. Write all raw output and a short summary (max CPU-vs-OpenCL and CPU-vs-Slang mean diff per
   case, before vs after; CPU timing before vs after) to
   /mnt/project-files/upstream-fixes/color-picker-double-sums/m4-results.md.
   If saving to project files from the Mac only works at the root, save there as
   m4-picker-double-sums.md.
