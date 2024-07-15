const builtin = @import("builtin");
const std = @import("std");

const c = @import("c.zig");
const shaderc = @import("shaderc.zig");
const ft = @import("ft.zig");

const Blit = @import("blit.zig").Blit;
const Preview = @import("preview.zig").Preview;
const Shader = @import("shaderc.zig").Shader;

const AsyncContext = struct {
    data: *anyopaque,
    event: *std.Thread.ResetEvent,
};

pub const Renderer = struct {
    const Self = @This();

    instance: c.WGPUInstanceDescriptor,
    adapter: c.WGPUAdapter,

    tex: c.WGPUTexture,
    tex_view: c.WGPUTextureView,
    tex_sampler: c.WGPUSampler,

    width: u32,
    height: u32,

    device: c.WGPUDevice,
    surface: c.WGPUSurface,

    queue: c.WGPUQueue,

    bind_group: c.WGPUBindGroup,
    uniform_buffer: c.WGPUBuffer,
    char_grid_buffer: c.WGPUBuffer,

    render_pipeline: c.WGPURenderPipeline,

    preview: ?*Preview,
    blit: Blit,

    // We track the last few preview times; if the media is under 30 FPS,
    // then we switch to tiled rendering
    dt: [5]i64,
    dt_index: usize,

    pub fn init(alloc: std.mem.Allocator, window: *c.GLFWwindow, atlas: *const ft.Atlas) !Self {
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();

        const desc: c.WGPUInstanceDescriptor = .{};
        const instance = c.wgpuCreateInstance(&desc);

        // Extract the WGPU Surface from the platform-specific window
        const platform = builtin.os.tag;
        const surface = switch (builtin.os.tag) {
            .macos => macos: {
                // Time to do hilarious Objective-C runtime hacks, equivalent to
                //  [ns_window.contentView setWantsLayer:YES];
                //  id metal_layer = [CAMetalLayer layer];
                //  [ns_window.contentView setLayer:metal_layer];
                const objc = @import("objc.zig");
                const darwin = @import("darwin.zig");

                const cocoa_window = darwin.glfwGetCocoaWindow(window);
                const ns_window: c.id = @ptrCast(@alignCast(cocoa_window));

                const cv = objc.call(ns_window, "contentView");
                _ = objc.call_(cv, "setWantsLayer:", true);

                const ca_metal = objc.class("CAMetalLayer");
                const metal_layer = objc.call(ca_metal, "layer");

                _ = objc.call_(cv, "setLayer:", metal_layer);

                break :macos c.wgpuInstanceCreateSurface(
                    instance,
                    &.{
                        .nextInChain = @ptrCast(&c.WGPUSurfaceDescriptorFromMetalLayer{
                            .chain = .{
                                .sType = c.WGPUSType_SurfaceDescriptorFromMetalLayer,
                            },
                            .layer = metal_layer,
                        }),
                    },
                );
            },
            else => {
                std.debug.panic("Unimplemented on platform {}", .{platform});
            },
        };

        ////////////////////////////////////////////////////////////////////////////
        // WGPU initial setup
        var adapter: c.WGPUAdapter = null;

        {
            var event: std.Thread.ResetEvent = .{};
            var ctx: AsyncContext = .{
                .data = @ptrCast(&adapter),
                .event = &event,
            };

            c.wgpuInstanceRequestAdapter(
                instance,
                &.{
                    .powerPreference = c.WGPUPowerPreference_HighPerformance,
                    .compatibleSurface = surface,
                },
                handleRequestAdapter,
                &ctx,
            );

            event.wait();
        }

        var device: c.WGPUDevice = null;

        {
            var event: std.Thread.ResetEvent = .{};
            var ctx: AsyncContext = .{
                .data = @ptrCast(&device),
                .event = &event,
            };

            c.wgpuAdapterRequestDevice(
                adapter,
                &.{
                    .requiredLimits = &.{
                        .limits = .{
                            .maxBindGroups = 1,
                        },
                    },
                },
                handleRequestDevice,
                &ctx,
            );

            event.wait();
        }

        ////////////////////////////////////////////////////////////////////////////
        // Build the shaders using shaderc
        const vert_spv = try shaderc.build_shader_from_file(&arena, "shaders/grid.vert");
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

        const frag_spv = try shaderc.build_shader_from_file(&arena, "shaders/grid.frag");
        const frag_shader = c.wgpuDeviceCreateShaderModule(
            device,
            &.{
                .nextInChain = @ptrCast(&c.WGPUShaderModuleSPIRVDescriptor{
                    .chain = .{
                        .sType = c.WGPUSType_ShaderModuleSPIRVDescriptor,
                    },
                    .code = frag_spv.ptr,
                    .codeSize = @intCast(frag_spv.len),
                }),
            },
        );
        defer c.wgpuShaderModuleRelease(frag_shader);

        ////////////////////////////////////////////////////////////////////////////
        // Upload the font atlas texture
        const tex_size: c.WGPUExtent3D = .{
            .width = @intCast(atlas.tex_size),
            .height = @intCast(atlas.tex_size),
            .depthOrArrayLayers = 1,
        };

        const tex = c.wgpuDeviceCreateTexture(
            device,
            &.{
                .size = tex_size,
                .mipLevelCount = 1,
                .sampleCount = 1,
                .dimension = c.WGPUTextureDimension_D2,
                .format = c.WGPUTextureFormat_Rgba8Unorm,
                // SAMPLED tells wgpu that we want to use this texture in shaders
                // COPY_DST means that we want to copy data to this texture
                .usage = c.WGPUTextureUsage_SAMPLED | c.WGPUTextureUsage_COPY_DST,
                .label = "font_atlas",
            },
        );

        const tex_view = c.wgpuTextureCreateView(
            tex,
            &.{
                .label = "font_atlas_view",
                .dimension = c.WGPUTextureViewDimension_D2,
                .format = c.WGPUTextureFormat_Rgba8Unorm,
                .aspect = c.WGPUTextureAspect_All,
                .baseMipLevel = 0,
                .mipLevelCount = 1,
                .baseArrayLayer = 0,
                .arrayLayerCount = 1,
            },
        );

        const tex_sampler = c.wgpuDeviceCreateSampler(
            device,
            &.{
                .label = "font_atlas_sampler",
                .addressModeU = c.WGPUAddressMode_ClampToEdge,
                .addressModeV = c.WGPUAddressMode_ClampToEdge,
                .addressModeW = c.WGPUAddressMode_ClampToEdge,
                .magFilter = c.WGPUFilterMode_Linear,
                .minFilter = c.WGPUFilterMode_Nearest,
                .mipmapFilter = c.WGPUFilterMode_Nearest,
                .lodMinClamp = 0.0,
                .lodMaxClamp = std.math.floatMax(f32),
                .compare = c.WGPUCompareFunction_Undefined,
            },
        );

        ////////////////////////////////////////////////////////////////////////////
        // Uniform buffers
        const uniform_buffer = c.wgpuDeviceCreateBuffer(
            device,
            &.{
                .label = "Uniforms",
                .size = @sizeOf(c.fpUniforms),
                .usage = c.WGPUBufferUsage_UNIFORM | c.WGPUBufferUsage_COPY_DST,
                .mappedAtCreation = false,
            },
        );
        const char_grid_buffer = c.wgpuDeviceCreateBuffer(
            device,
            &.{
                .label = "Character grid",
                .size = @sizeOf(u32) * 512 * 512,
                .usage = c.WGPUBufferUsage_STORAGE | c.WGPUBufferUsage_COPY_DST,
                .mappedAtCreation = false,
            },
        );

        ////////////////////////////////////////////////////////////////////////////
        // Bind groups (?!)
        const bind_group_layout_entries = [_]c.WGPUBindGroupLayoutEntry{
            .{
                .binding = 0,
                .visibility = c.WGPUShaderStage_Fragment,
                .ty = c.WGPUBindingType_SampledTexture,
                .texture = .{
                    .sampleType = c.WGPUTextureSampleType_Uint,
                    .multisampled = false,
                },
                .storageTexture = .{
                    .format = c.WGPUTextureFormat_Rgba8Unorm,
                    .viewDimension = c.WGPUTextureViewDimension_D2,
                },
                .count = undefined,
                .has_dynamic_offset = undefined,
                .min_buffer_binding_size = undefined,
            },
            .{
                .binding = 1,
                .visibility = c.WGPUShaderStage_Fragment,
                .ty = c.WGPUBindingType_Sampler,
                .multisampled = undefined,
                .view_dimension = undefined,
                .texture_component_type = undefined,
                .storage_texture_format = undefined,
                .count = undefined,
                .has_dynamic_offset = undefined,
                .min_buffer_binding_size = undefined,
            },
            .{
                .binding = 2,
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
            .{
                .binding = 3,
                .visibility = c.WGPUShaderStage_VERTEX,
                .ty = c.WGPUBindingType_StorageBuffer,
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
                .entryCount = bind_group_layout_entries.len,
            },
        );
        defer c.wgpuBindGroupLayoutRelease(bind_group_layout);

        const bind_group_entries = [_]c.WGPUBindGroupEntry{
            .{
                .binding = 0,
                .texture_view = tex_view,
                .sampler = 0, // None
                .buffer = 0, // None
                .offset = undefined,
                .size = undefined,
            },
            .{
                .binding = 1,
                .sampler = tex_sampler,
                .texture_view = 0, // None
                .buffer = 0, // None
                .offset = undefined,
                .size = undefined,
            },
            .{
                .binding = 2,
                .buffer = uniform_buffer,
                .offset = 0,
                .size = @sizeOf(c.fpUniforms),
                .sampler = 0, // None
                .texture_view = 0, // None
            },
            .{
                .binding = 3,
                .buffer = char_grid_buffer,
                .offset = 0,
                .size = @sizeOf(u32) * 512 * 512,
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
        const bind_group_layouts = [_]c.WGPUBindGroupLayout{bind_group_layout};

        ////////////////////////////////////////////////////////////////////////////
        // Render pipelines (?!?)
        const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(
            device,
            &.{
                .bindGroupLayouts = &bind_group_layouts,
                .bindGroupLayoutCount = bind_group_layouts.len,
            },
        );
        defer c.wgpuPipelineLayoutRelease(pipeline_layout);

        const render_pipeline = c.wgpuDeviceCreateRenderPipeline(
            device,
            &.{
                .layout = pipeline_layout,
                .vertex = .{
                    .module = vert_shader,
                    .entry_point = "main",
                },
                .fragment = &.{
                    .module = frag_shader,
                    .entry_point = "main",
                },
                .rasterization_state = &.{
                    .front_face = c.WGPUFrontFace_Ccw,
                    .cull_mode = c.WGPUCullMode_None,
                    .depth_bias = 0,
                    .depth_bias_slope_scale = 0.0,
                    .depth_bias_clamp = 0.0,
                },
                .primitive_topology = c.WGPUPrimitiveTopology_TriangleList,
                .color_states = &.{
                    .format = c.WGPUTextureFormat_Bgra8Unorm,
                    .alpha_blend = .{
                        .src_factor = c.WGPUBlendFactor_One,
                        .dst_factor = c.WGPUBlendFactor_Zero,
                        .operation = c.WGPUBlendOperation_Add,
                    },
                    .color_blend = .{
                        .src_factor = c.WGPUBlendFactor_One,
                        .dst_factor = c.WGPUBlendFactor_Zero,
                        .operation = c.WGPUBlendOperation_Add,
                    },
                    .write_mask = c.WGPUColorWrite_ALL,
                },
                .color_states_length = 1,
                .depth_stencil_state = null,
                .vertex_state = .{
                    .index_format = c.WGPUIndexFormat_Uint16,
                    .vertex_buffers = null,
                    .vertex_buffers_length = 0,
                },
                .sample_count = 1,
                .sample_mask = 0,
                .alpha_to_coverage_enabled = false,
            },
        );

        var out: Renderer = .{
            .instance = instance,
            .tex = tex,
            .tex_view = tex_view,
            .tex_sampler = tex_sampler,

            .swap_chain = undefined, // assigned in resize_swap_chain below
            .width = undefined,
            .height = undefined,

            .device = device,
            .surface = surface,

            .queue = c.wgpuDeviceGetQueue(device),

            .bind_group = bind_group,
            .uniform_buffer = uniform_buffer,
            .char_grid_buffer = char_grid_buffer,

            .render_pipeline = render_pipeline,

            .preview = null,
            .blit = try Blit.init(alloc, device),

            .dt = undefined,
            .dt_index = 0,
        };

        out.reset_dt();
        out.update_font_tex(atlas);
        return out;
    }

    pub fn clear_preview(self: *Self, alloc: std.mem.Allocator) void {
        if (self.preview) |p| {
            p.deinit();
            alloc.destroy(p);
            self.preview = null;
        }
    }

    fn reset_dt(self: *Self) void {
        var i: usize = 0;
        while (i < self.dt.len) : (i += 1) {
            self.dt[i] = 0;
        }
        self.dt_index = 0;
    }

    pub fn update_preview(self: *Self, alloc: std.mem.Allocator, s: Shader) !void {
        self.clear_preview(alloc);

        // Construct a new Preview with our current state
        var p = try alloc.create(Preview);
        p.* = try Preview.init(alloc, self.device, s.spirv, s.has_time);
        p.set_size(self.width, self.height);

        self.preview = p;
        self.blit.bind_to_tex(p.tex_view[1]);
        self.reset_dt();
    }

    pub fn update_font_tex(self: *Self, font: *const ft.Atlas) void {
        const tex_size: c.WGPUExtent3D = .{
            .width = @intCast(font.tex_size),
            .height = @intCast(font.tex_size),
            .depthOrArrayLayers = 1,
        };
        c.wgpuQueueWriteTexture(
            self.queue,
            &.{
                .texture = self.tex,
                .mip_level = 0,
                .origin = (c.WGPUOrigin3D){ .x = 0, .y = 0, .z = 0 },
            },
            @ptrCast(font.tex.ptr),
            font.tex.len * @sizeOf(u32),
            &.{
                .offset = 0,
                .bytes_per_row = @as(u32, @intCast(font.tex_size)) * @sizeOf(u32),
                .rows_per_image = @as(u32, @intCast(font.tex_size)) * @sizeOf(u32),
            },
            &tex_size,
        );
    }

    pub fn redraw(self: *Self, total_tiles: u32) void {
        const start_ms = std.time.milliTimestamp();

        // Render the preview to its internal texture, then blit from that
        // texture to the main swap chain.  This lets us render the preview
        // at a different resolution from the rest of the UI.
        if (self.preview) |p| {
            p.redraw();
            if ((p.uniforms._tiles_per_side > 1 and p.uniforms._tile_num != 0) or
                p.draw_continuously)
            {
                c.glfwPostEmptyEvent();
            }
        }

        // Begin the main render operation
        const next_texture = c.wgpuSwapChainGetNextTexture(self.swap_chain);
        if (next_texture.view_id == 0) {
            std.debug.panic("Cannot acquire next swap chain texture", .{});
        }

        const cmd_encoder = c.wgpuDeviceCreateCommandEncoder(
            self.device,
            &(c.WGPUCommandEncoderDescriptor){ .label = "main encoder" },
        );

        const color_attachments = [_]c.WGPURenderPassColorAttachmentDescriptor{
            .{
                .attachment = next_texture.view_id,
                .resolve_target = 0,
                .channel = .{
                    .load_op = c.WGPULoadOp_Clear,
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
        c.wgpuRenderPassEncoderDraw(rpass, total_tiles * 6, 1, 0, 0);
        if (self.preview != null) {
            self.blit.redraw(next_texture, cmd_encoder);
        }

        const cmd_buf = c.wgpuCommandEncoderFinish(cmd_encoder, null);
        c.wgpuQueueSubmit(self.queue, &cmd_buf, 1);

        c.wgpuSwapChainPresent(self.swap_chain);

        const end_ms = std.time.milliTimestamp();
        self.dt[self.dt_index] = end_ms - start_ms;
        self.dt_index = (self.dt_index + 1) % self.dt.len;

        var dt_local = self.dt;
        const asc = std.sort.asc(i64);
        std.mem.sort(i64, dt_local[0..], {}, asc);
        const dt = dt_local[self.dt.len / 2];

        if (dt > 33) {
            if (self.preview) |p| {
                p.adjust_tiles(dt);
                self.reset_dt();
            }
        }
    }

    pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
        c.wgpuInstanceRelease(self.instance);
        c.wgpuTextureRelease(self.tex);
        c.wgpuTextureViewRelease(self.tex_view);
        c.wgpuSamplerRelease(self.tex_sampler);

        c.wgpuBindGroupRelease(self.bind_group);
        c.wgpuBufferRelease(self.uniform_buffer);
        c.wgpuBufferRelease(self.char_grid_buffer);

        c.wgpuRenderPipelineRelease(self.render_pipeline);

        if (self.preview) |p| {
            p.deinit();
            alloc.destroy(p);
        }
        self.blit.deinit();
    }

    pub fn update_grid(self: *Self, char_grid: []u32) void {
        c.wgpuQueueWriteBuffer(
            self.queue,
            self.char_grid_buffer,
            0,
            @ptrCast(char_grid.ptr),
            char_grid.len * @sizeOf(u32),
        );
    }

    pub fn resize_swap_chain(self: *Self, width: u32, height: u32) void {
        self.swap_chain = c.wgpuDeviceCreateSwapChain(
            self.device,
            self.surface,
            &.{
                .usage = c.WGPUTextureUsage_OUTPUT_ATTACHMENT,
                .format = c.WGPUTextureFormat_Bgra8Unorm,
                .width = width,
                .height = height,
                .present_mode = c.WGPUPresentMode_Fifo,
            },
        );

        // Track width and height so that we can set them in a Preview
        // (even if one isn't loaded right now)
        self.width = width;
        self.height = height;
        if (self.preview) |p| {
            p.set_size(width, height);
            self.blit.bind_to_tex(p.tex_view[1]);
        }
    }

    pub fn update_uniforms(self: *Self, u: *const c.fpUniforms) void {
        c.wgpuQueueWriteBuffer(
            self.queue,
            self.uniform_buffer,
            0,
            @ptrCast(u),
            @sizeOf(c.fpUniforms),
        );
    }
};

export fn handleRequestAdapter(status: c.WGPURequestAdapterStatus, received: c.WGPUAdapter, _: [*c]const u8, userdata: ?*anyopaque) void {
    std.debug.assert(status == c.WGPURequestAdapterStatus_Success);
    const ctx: *AsyncContext = @ptrCast(@alignCast(userdata));
    const adapter: *c.WGPUAdapter = @ptrCast(@alignCast(ctx.data));
    adapter.* = received;
    ctx.event.set();
}

export fn handleRequestDevice(status: c.WGPURequestDeviceStatus, received: c.WGPUDevice, _: [*c]const u8, userdata: ?*anyopaque) void {
    std.debug.assert(status == c.WGPURequestDeviceStatus_Success);
    const ctx: *AsyncContext = @ptrCast(@alignCast(userdata));
    const device: *c.WGPUDevice = @ptrCast(@alignCast(ctx.data));
    device.* = received;
    ctx.event.set();
}
