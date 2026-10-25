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

#pragma once

#include <math.h>
#include <float.h>  // FLT_MAX

#include "common/fast_guided_filter.h" // interpolate_bilinear
#include "common/gaussian.h"           // dt_gaussian_*

// Guided filter surface gaussian blurring with edge preservation.
// supports negative values for log-luminance data (any sign).
// Blurs in-place on a downscaled grey image.
__DT_CLONE_TARGETS__
static inline void fast_gaussian_guided_filter(float *const restrict image,
                                               const size_t width,
                                               const size_t height,
                                               const float sigma,
                                               const float feathering)
{
  const float scaling = fmaxf(fminf(sigma, 4.0f), 1.0f);
  const float downsampled_sigma = fmaxf(sigma / scaling, 1.0f);

  const size_t downsampled_height = height / scaling;
  const size_t downsampled_width = width / scaling;
  const size_t num_elem_ds = downsampled_width * downsampled_height;
  const size_t num_elem = width * height;

  float *const restrict downsampled_image = dt_alloc_align_float(num_elem_ds);
  float *const restrict gaussian_input = dt_alloc_align_float(num_elem_ds * 2);
  // Statistics for the guide; average and variance; downsampled and full resolution versions.
  float *const restrict downsampled_statistics = dt_alloc_align_float(num_elem_ds * 2);
  float *const restrict full_resolution_statistics = dt_alloc_align_float(num_elem * 2);

  if(!downsampled_image || !gaussian_input || !downsampled_statistics || !full_resolution_statistics)
  {
    dt_control_log(_("fast gaussian guided filter failed to allocate memory, check your RAM settings"));
    goto clean;
  }

  interpolate_bilinear(image, width, height, downsampled_image, downsampled_width, downsampled_height, 1);

  // channels: value and squared value, with ranges for the gaussian
  float ming = FLT_MAX;
  float maxg = -FLT_MAX;
  float maxg2 = 0.0f;
  DT_OMP_FOR(reduction(max:maxg, maxg2) reduction(min:ming))
  for(size_t k = 0; k < num_elem_ds; k++)
  {
    const float p = downsampled_image[k];
    gaussian_input[2 * k] = p;
    gaussian_input[2 * k + 1] = p * p;
    ming = fminf(ming, p);
    maxg = fmaxf(maxg, p);
    maxg2 = fmaxf(maxg2, p * p);
  }
  // squares of values spanning zero have a minimum of 0
  const float ming2 = (ming <= 0.0f && maxg >= 0.0f) ? 0.0f
                      : fminf(ming * ming, maxg * maxg);

  const float channel_max[2] = { maxg, maxg2 };
  const float channel_min[2] = { ming, ming2 };
  dt_gaussian_t *g = dt_gaussian_init(downsampled_width, downsampled_height, 2, channel_max, channel_min, downsampled_sigma, 0);
  if(!g) goto clean;
  dt_gaussian_blur(g, gaussian_input, downsampled_statistics);
  dt_gaussian_free(g);

  interpolate_bilinear(downsampled_statistics, downsampled_width, downsampled_height,
                       full_resolution_statistics, width, height, 2);

  DT_OMP_FOR()
  for(size_t k = 0; k < num_elem; k++)
  {
    const float average = full_resolution_statistics[2 * k];
    const float variance = fmaxf(full_resolution_statistics[2 * k + 1] - average * average, 0.0f);
    const float a = variance / (variance + feathering);
    image[k] = a * image[k] + (1.0f - a) * average;
  }

clean:
  dt_free_align(full_resolution_statistics);
  dt_free_align(downsampled_statistics);
  dt_free_align(gaussian_input);
  dt_free_align(downsampled_image);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on