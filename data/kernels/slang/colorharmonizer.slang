/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/


// Slang port of ../colorharmonizer.cl

import common;
import colorspace;

float _wrap_hue(float h)
{
  h = fmod(h, 1.0f);
  if(h < 0.0f) h += 1.0f;
  return h;
}

static const int MAX_NODES = 4;
static const float BOUNDARY_BAND = 0.01f;
static const float GAMUT_FLT_MIN = 1.175494350822287507969e-38f;

// pull the pixel hue toward the nearest harmony node, scaled by gaussian
// proximity, and return the matching saturation delta; candidates within
// BOUNDARY_BAND of the nearest node are blended so the shift ramps across
// zone boundaries, see get_weighted_hue_shift() in src/iop/colorharmonizer.c
float2 _get_weighted_hue_shift(const float px_hue, StructuredBuffer<float> nodes,
                               StructuredBuffer<float> node_saturation, const int num_nodes,
                               const float zone_width_factor)
{
  if(num_nodes <= 0) return float2(0.0f, 0.0f);

  const float sigma = fp32_div(zone_width_factor * 0.5f, float(num_nodes));
  const float inv_2sigma2 = fp32_div(1.0f, 2.0f * sigma * sigma);

  float diff[2 * MAX_NODES];
  int n = 0;
  float d_min = 1.0f;
  for(int i = 0; i < num_nodes; i++)
  {
    float s = nodes[i] - px_hue;
    if(s > 0.5f) s -= 1.0f;
    else if(s < -0.5f) s += 1.0f;
    diff[n++] = s;
    diff[n++] = s - copysign(1.0f, s);
    d_min = min(d_min, abs(s));
  }

  float sum_a = 0.0f, shift = 0.0f, sat = 0.0f;
  for(int k = 0; k < n; k++)
  {
    const float d = abs(diff[k]);
    const float a = 1.0f - fp32_div(d - d_min, BOUNDARY_BAND);
    if(a <= 0.0f) continue;
    const float w = exp(-d * d * inv_2sigma2);
    sum_a += a;
    shift += a * diff[k] * w;
    sat += a * (node_saturation[k / 2] - 1.0f) * w;
  }

  return float2(fp32_div(shift, sum_a), fp32_div(sat, sum_a));
}

// cap the shifted chroma at the working gamut boundary for the new hue, see
// _gamut_limit_chroma() in src/iop/colorharmonizer.c; hues in radians
float _gamut_limit_chroma(StructuredBuffer<float> gamut_LUT, const float J, const float chroma_in,
                          const float hue_in, const float chroma_out, const float hue_out,
                          const float L_white)
{
  const float k = fp32_div(15.932993652962535f * pow(J * L_white, 0.6523997524738018f), L_white);
  const float g_in = pow(max(lookup_gamut(gamut_LUT, hue_in), GAMUT_FLT_MIN), 0.6007557017508491f);
  const float g_out = pow(max(lookup_gamut(gamut_LUT, hue_out), GAMUT_FLT_MIN), 0.6007557017508491f);
  return min(chroma_out, g_out * max(k, fp32_div(chroma_in, g_in)));
}

// the per-pixel corrections (hue_delta, sat_delta), and the forward JCH
// conversion as (J, chroma, normalized hue, alpha) for the apply kernel
[shader("compute")]
[numthreads(16, 16, 1)]
void colorharmonizer_map(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform RWStructuredBuffer<float2> p_out,
                         uniform RWStructuredBuffer<float4> jch_out, uniform int width, uniform int height,
                         uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> nodes,
                         uniform int num_nodes, uniform float zone_width,
                         uniform StructuredBuffer<float> node_saturation, uniform float L_white)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pix_in = in.Areadpixel(x, y);
  const float4 XYZ_D65 = matrix_product_float4(max(0.0f, pix_in), matrix_in);
  const float4 xyY = dt_D65_XYZ_to_xyY(XYZ_D65);
  const float4 JCH = xyY_to_dt_UCS_JCH(xyY, L_white);

  const float hue = fp32_div(JCH.z + M_PI_F, DT_2PI_F);

  const int idx = y * width + x;
  jch_out[idx] = float4(JCH.x, JCH.y, hue, pix_in.w);

  p_out[idx] = _get_weighted_hue_shift(hue, nodes, node_saturation, num_nodes, zone_width);
}

// the smoothed corrections applied to the cached JCH
[shader("compute")]
[numthreads(16, 16, 1)]
void colorharmonizer_apply(uint3 gid : SV_DispatchThreadID,
                           uniform ImageOut out, uniform int width, uniform int height,
                           uniform StructuredBuffer<float> matrix_out, uniform StructuredBuffer<float4> jch_in,
                           uniform StructuredBuffer<float2> corrections, uniform float effect_strength,
                           uniform float protect_neutral, uniform StructuredBuffer<float> gamut_LUT,
                           uniform float L_white)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float4 cached = jch_in[k];
  const float chroma = cached.y;
  const float2 corr = corrections[k];

  const float t = protect_neutral;
  const float cutoff = t * t * t * 0.03f;
  const float chroma_weight = fp32_div(chroma, chroma + cutoff + 1.0e-5f);

  const float hue_out = _wrap_hue(cached.z + corr.x * effect_strength * chroma_weight) * DT_2PI_F - M_PI_F;
  const float chroma_out = _gamut_limit_chroma(gamut_LUT, cached.x, chroma, cached.z * DT_2PI_F - M_PI_F,
                                               max(chroma * (1.0f + corr.y * chroma_weight), 0.0f),
                                               hue_out, L_white);
  const float4 JCH = float4(cached.x, chroma_out, hue_out, 0.0f);

  const float4 xyY = dt_UCS_JCH_to_xyY(JCH, L_white);
  float4 pix_out = matrix_product_float4(dt_xyY_to_XYZ(xyY), matrix_out);
  pix_out.w = cached.w;
  out.write(int2(x, y), pix_out);
}
