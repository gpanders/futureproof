const std = @import("std");

const c = @import("c.zig");
const shaderc = @import("shaderc.zig");

pub const Preview = struct {
    const Self = @This();

    device: c.WGPUDevice,
    queue: c.WGPUQueue,

    // We render into tex[0] in tiles to keep up a good framerate, then
    // copy to tex[1] to render the complete image without tearing
    tex: [2]c.WGPUTexture,
    tex_view: [2]c.WGPUTextureView,
    tex_size: c.WGPUExtent3D,

    bind_group: c.WGPUBindGroup,
    uniform_buffer: c.WGPUBuffer,
    render_pipeline: c.WGPURenderPipeline,

    start_time: i64,
    uniforms: c.fpPreviewUniforms,
    draw_continuously: bool,

    pub fn init(
        alloc: std.mem.Allocator,
        device: c.WGPUDevice,
        frag: []const u32,
        draw_continuously: bool,
    ) !Preview {
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();

        // Build the shaders using shaderc
        const vert_spv = shaderc.build_shader_from_file(&arena, "shaders/preview.vert") catch {
            std.debug.panic("Could not build preview.vert", .{});
        };
        const vert_shader = c.wgpuDeviceCreateShaderModule(
            device,
            &.{
                .nextInChain = @ptrCast(&c.WGPUShaderModuleSPIRVDescriptor{
                    .chain = .{
                        .sType = c.WGPUSType_ShaderModuleSPIRVDescriptor,
                    },
                    .code = vert_spv.ptr,
                    .codeSize = @intCast(vert_spv.len),
                }),
            },
        );
        defer c.wgpuShaderModuleRelease(vert_shader);

        const frag_shader = c.wgpuDeviceCreateShaderModule(
            device,
            &.{
                .nextInChain = @ptrCast(&c.WGPUShaderModuleSPIRVDescriptor{
                    .chain = .{
                        .sType = c.WGPUSType_ShaderModuleSPIRVDescriptor,
                    },
                    .code = frag.ptr,
                    .codeSize = @intCast(frag.len),
                }),
            },
        );
        defer c.wgpuShaderModuleRelease(frag_shader);

        ////////////////////////////////////////////////////////////////////////////////
        // Uniform buffers
        const uniform_buffer = c.wgpuDeviceCreateBuffer(
            device,
            &.{
                .label = "Uniforms",
                .size = @sizeOf(c.fpPreviewUniforms),
                .usage = c.WGPUBufferUsage_UNIFORM | c.WGPUBufferUsage_COPY_DST,
                .mapped_at_creation = false,
            },
        );

        ////////////////////////////////////////////////////////////////////////////////
        const bind_group_layout_entries = [_]c.WGPUBindGroupLayoutEntry{
            .{
                .binding = 0,
                .visibility = c.WGPUShaderStage_VERTEX | c.WGPUShaderStage_FRAGMENT,
                .ty = c.WGPUBindingType_UniformBuffer,

                .has_dynamic_offset = false,
                .min_buffer_binding_size = 0,

                .multisampled = undefined,
                .view_dimension = undefined,
                .texture_component_type = undefined,
                .storage_texture_format = undefined,
                .count = undefined,
            },
        };
        const bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(
            device,
            &.{
                .label = "bind group layout",
                .entries = &bind_group_layout_entries,
                .entries_length = bind_group_layout_entries.len,
            },
        );
        defer c.wgpuBindGroupLayoutRelease(bind_group_layout);
        const bind_group_entries = [_]c.WGPUBindGroupEntry{
            .{
                .binding = 0,
                .buffer = uniform_buffer,
                .offset = 0,
                .size = @sizeOf(c.fpPreviewUniforms),

                .sampler = 0, // None
                .texture_view = 0, // None
            },
        };
        const bind_group = c.wgpuDeviceCreateBindGroup(
            device,
            &.{
                .label = "bind group",
                .layout = bind_group_layout,
                .entries = &bind_group_entries,
                .entries_length = bind_group_entries.len,
            },
        );
        const bind_group_layouts = [_]c.WGPUBindGroup{bind_group_layout};

        // Render pipelines (?!?)
        const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(
            device,
            &.{
                .bind_group_layouts = &bind_group_layouts,
                .bind_group_layouts_length = bind_group_layouts.len,
            },
        );
        defer c.wgpuPipelineLayoutRelease(pipeline_layout);

        const render_pipeline = c.wgpuDeviceCreateRenderPipeline(
            device,
            &.{
                .layout = pipeline_layout,
                .vertex = .{
                    .module = vert_shader,
                    .entryPoint = "main",
                },
                .fragment = &.{
                    .module = frag_shader,
                    .entryPoint = "main",
                    .targetCount = 1,
                    .targets = &.{
                        .format = c.WGPUTextureFormat_Bgra8Unorm,
                        .blend = &.{
                            .color = .{
                                .operation = c.WGPUBlendOperation_Add,
                                .srcFactor = c.WGPUBlendFactor_One,
                                .dstFactor = c.WGPUBlendFactor_Zero,
                            },
                            .alpha = .{
                                .operation = c.WGPUBlendOperation_Add,
                                .srcFactor = c.WGPUBlendFactor_One,
                                .dstFactor = c.WGPUBlendFactor_Zero,
                            },
                        },
                        .writeMask = c.WGPUColorWrite_ALL,
                    },
                },
                .depthStencil = &.{
                    .format = c.WGPUTextureFormat_Bgra8Unorm,
                    .depthBias = 0,
                    .depthBiasSlopeScale = 0.0,
                    .depthBiasClamp = 0.0,
                },
                .primitive = .{
                    .topology = c.WGPUPrimitiveTopology_TriangleList,
                    .stripIndexFormat = c.WGPUIndexFormat_Uint16,
                    .frontFace = c.WGPUFrontFace_Ccw,
                    .cullMode = c.WGPUCullMode_None,
                },
                .multisample = .{
                    .count = 1,
                    .mask = 0,
                    .alphaToCoverageEnabled = false,
                },
            },
        );

        const start_time = std.time.milliTimestamp();
        return Self{
            .device = device,
            .queue = c.wgpuDeviceGetDefaultQueue(device),

            .render_pipeline = render_pipeline,
            .uniform_buffer = uniform_buffer,
            .bind_group = bind_group,

            .start_time = start_time,
            .draw_continuously = draw_continuously,

            // Assigned in set_size below
            .tex = undefined,
            .tex_view = undefined,
            .tex_size = undefined,

            .uniforms = .{
                .iResolution = .{ .x = 0, .y = 0, .z = 0 },
                .iTime = 0.0,
                .iMouse = .{ .x = 0, .y = 0, .z = 0, .w = 0 },
                ._tiles_per_side = 1,
                ._tile_num = 0,
            },
        };
    }

    fn destroy_textures(self: *const Self) void {
        // If the texture was created, then destroy it
        if (self.uniforms.iResolution.x != 0) {
            for (self.tex) |t| {
                c.wgpuTextureRelease(t);
            }
            for (self.tex_view) |t| {
                c.wgpuTextureViewRelease(t);
            }
        }
    }

    pub fn adjust_tiles(self: *Self, dt: i64) void {
        // What's the total render time, approximately?
        const dt_est = std.math.pow(i64, self.uniforms._tiles_per_side, 2) * dt;

        // We'd like to keep the UI running at 60 FPS, approximately
        const t = std.math.ceil(std.math.sqrt(@as(f32, @floatFromInt(@divFloor(dt_est, 16)))));

        std.debug.print(
            "Switching from {} to {} tiles per side\n",
            .{ self.uniforms._tiles_per_side, t },
        );
        var t_: u32 = @intFromFloat(t);
        if (t_ > 5) {
            t_ = 5;
        }
        self.uniforms._tiles_per_side = t_;
        self.uniforms._tile_num = 0;
    }

    pub fn deinit(self: *const Self) void {
        c.wgpuBindGroupRelease(self.bind_group);
        c.wgpuBufferRelease(self.uniform_buffer);
        c.wgpuRenderPipelineRelease(self.render_pipeline);
        self.destroy_textures();
    }

    pub fn set_size(self: *Self, width: u32, height: u32) void {
        self.destroy_textures();

        self.tex_size = .{
            .width = @intCast(width / 2),
            .height = @intCast(height),
            .depthOrArrayLayers = 1,
        };

        var i: u8 = 0;
        while (i < 2) : (i += 1) {
            self.tex[i] = c.wgpuDeviceCreateTexture(
                self.device,
                &.{
                    .size = self.tex_size,
                    .mip_level_count = 1,
                    .sample_count = 1,
                    .dimension = c.WGPUTextureDimension_D2,
                    .format = c.WGPUTextureFormat_Bgra8Unorm,

                    // We render to this texture, then use it as a source when
                    // blitting into the final UI image
                    .usage = if (i == 0)
                        (c.WGPUTextureUsage_OUTPUT_ATTACHMENT |
                            c.WGPUTextureUsage_COPY_SRC)
                    else
                        (c.WGPUTextureUsage_OUTPUT_ATTACHMENT |
                            c.WGPUTextureUsage_COPY_SRC |
                            c.WGPUTextureUsage_SAMPLED |
                            c.WGPUTextureUsage_COPY_DST),
                    .label = "preview_tex",
                },
            );

            self.tex_view[i] = c.wgpuTextureCreateView(
                self.tex[i],
                &.{
                    .label = "preview_tex_view",
                    .dimension = c.WGPUTextureViewDimension_D2,
                    .format = c.WGPUTextureFormat_Bgra8Unorm,
                    .aspect = c.WGPUTextureAspect_All,
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .array_layer_count = 1,
                },
            );
        }

        self.uniforms.iResolution.x = @as(f32, @floatFromInt(width)) / 2;
        self.uniforms.iResolution.y = @floatFromInt(height);
    }

    pub fn redraw(self: *Self) void {
        const cmd_encoder = c.wgpuDeviceCreateCommandEncoder(
            self.device,
            &.{ .label = "preview encoder" },
        );

        // Set the time in the uniforms array
        if (self.uniforms._tile_num == 0) {
            const time_ms = std.time.milliTimestamp() - self.start_time;
            self.uniforms.iTime = @as(f32, @floatFromInt(time_ms)) / 1000.0;
        }

        c.wgpuQueueWriteBuffer(
            self.queue,
            self.uniform_buffer,
            0,
            @ptrCast(&self.uniforms),
            @sizeOf(c.fpPreviewUniforms),
        );

        const load_op = if (self.uniforms._tile_num == 0)
            c.WGPULoadOp_Clear
        else
            c.WGPULoadOp_Load;
        const color_attachments = [_]c.WGPURenderPassColorAttachmentDescriptor{
            .{
                .attachment = if (self.uniforms._tiles_per_side == 1) self.tex_view[1] else self.tex_view[0],
                .resolve_target = 0,
                .channel = .{
                    .load_op = @intCast(load_op),
                    .store_op = c.WGPUStoreOp_Store,
                    .clear_value = .{
                        .r = 0.0,
                        .g = 0.0,
                        .b = 0.0,
                        .a = 1.0,
                    },
                    .read_only = false,
                },
            },
        };

        const rpass = c.wgpuCommandEncoderBeginRenderPass(
            cmd_encoder,
            &.{
                .color_attachments = &color_attachments,
                .color_attachments_length = color_attachments.len,
                .depth_stencil_attachment = null,
            },
        );

        c.wgpuRenderPassEncoderSetPipeline(rpass, self.render_pipeline);
        c.wgpuRenderPassEncoderSetBindGroup(rpass, 0, self.bind_group, null, 0);
        c.wgpuRenderPassEncoderDraw(rpass, 6, 1, 0, 0);

        // Move on to the next tile
        if (self.uniforms._tiles_per_side > 1) {
            self.uniforms._tile_num += 1;
        }

        // If we just finished rendering every tile, then also copy
        // to the deployment tex
        if (self.uniforms._tile_num == std.math.pow(u32, self.uniforms._tiles_per_side, 2)) {
            const src: c.WGPUTextureCopyView = .{
                .texture = self.tex[0],
                .mip_level = 0,
                .origin = .{ .x = 0, .y = 0, .z = 0 },
            };
            const dst: c.WGPUTextureCopyView = .{
                .texture = self.tex[1],
                .mip_level = 0,
                .origin = .{ .x = 0, .y = 0, .z = 0 },
            };
            c.wgpuCommandEncoderCopyTextureToTexture(
                cmd_encoder,
                &src,
                &dst,
                &self.tex_size,
            );
            self.uniforms._tile_num = 0;
        }

        const cmd_buf = c.wgpuCommandEncoderFinish(cmd_encoder, null);
        c.wgpuQueueSubmit(self.queue, &cmd_buf, 1);
    }
};
