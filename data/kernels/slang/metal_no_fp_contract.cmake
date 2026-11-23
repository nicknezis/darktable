# prepends the Metal pragma that turns off fma contraction to the Metal
# source in FILE, for kernels compiled at run time, where MTLCompileOptions
# has no switch for it. The .metallib builds pass -ffp-contract=off instead
file(READ ${FILE} DT_METAL_SOURCE)
file(WRITE ${FILE} "#pragma METAL fp contract(off)\n${DT_METAL_SOURCE}")
