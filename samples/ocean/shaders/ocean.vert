/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coords;

layout(location = 0) out vec2 frag_tex_coord;
layout(location = 1) out vec4 ec_pos;

layout(set = 0, binding = 0) uniform sampler2D u_displacement_map;
layout(std140, set = 0, binding = 2) uniform ViewData
{
    mat4 view_mat;
    mat4 proj_mat;
    vec3 sun_dir;
    float choppiness;
    float alt_scale;
}
view;

void main()
{
    // the displacement map is computed by the SYCL simulation
    vec3 displ = texture(u_displacement_map, in_tex_coords).rbg;
    displ.xy *= view.choppiness;
    displ.z *= view.alt_scale;

    vec3 ocean_vert = in_position + displ;
    ec_pos = view.view_mat * vec4(ocean_vert, 1.0);
    gl_Position = view.proj_mat * ec_pos;
    frag_tex_coord = in_tex_coords;
}
