const std = @import("std");

const c = @import("c.zig");
const shaderc = @import("shaderc.zig");

pub const Blit = struct {
    const Self = @This();

    device: c.WGPUDevice,

    bind_group_layout: c.WGPUBindGroupLayout,
    tex_sampler: c.WGPUSampler,
    bind_group: ?c.WGPUBindGroup,

    render_pipeline: c.WGPURenderPipeline,

    pub fn init(alloc: std.mem.Allocator, device: c.WGPUDevice) !Blit {
        var arena = std.heap.ArenaAllocator.init(alloc);
        const tmp_alloc = arena.allocator();
        defer arena.deinit();

        ////////////////////////////////////////////////////////////////////////////
        // Build the shaders using shaderc
        const vert_spv = shaderc.build_shader_from_file(tmp_alloc, "shaders/blit.vert") catch {
            std.debug.panic("Could not open file", .{});
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

        const frag_spv = shaderc.build_shader_from_file(tmp_alloc, "shaders/blit.frag") catch {
            std.debug.panic("Could not open file", .{});
        };
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

        ///////////////////////////////////////////////////////////////////////
        // Texture sampler (the texture comes from the Preview struct)
        const tex_sampler = c.wgpuDeviceCreateSampler(device, &(c.WGPUSamplerDescriptor){
            .next_in_chain = null,
            .label = "font_atlas_sampler",
            .address_mode_u = c.WGPUAddressMode_ClampToEdge,
            .address_mode_v = c.WGPUAddressMode_ClampToEdge,
            .address_mode_w = c.WGPUAddressMode_ClampToEdge,
            .mag_filter = c.WGPUFilterMode_Linear,
            .min_filter = c.WGPUFilterMode_Nearest,
            .mipmap_filter = c.WGPUFilterMode_Nearest,
            .lod_min_clamp = 0.0,
            .lod_max_clamp = std.math.floatMax(f32),
            .compare = c.WGPUCompareFunction_Undefined,
        });

        ////////////////////////////////////////////////////////////////////////////
        // Bind groups (?!)
        const bind_group_layout_entries = [_]c.WGPUBindGroupLayoutEntry{
            (c.WGPUBindGroupLayoutEntry){
                .binding = 0,
                .visibility = c.WGPUShaderStage_FRAGMENT,
                .ty = c.WGPUBindingType_SampledTexture,

                .multisampled = false,
                .view_dimension = c.WGPUTextureViewDimension_D2,
                .texture_component_type = c.WGPUTextureComponentType_Uint,
                .storage_texture_format = c.WGPUTextureFormat_Bgra8Unorm,

                .count = undefined,
                .has_dynamic_offset = undefined,
                .min_buffer_binding_size = undefined,
            },
            (c.WGPUBindGroupLayoutEntry){
                .binding = 1,
                .visibility = c.WGPUShaderStage_FRAGMENT,
                .ty = c.WGPUBindingType_Sampler,

                .multisampled = undefined,
                .view_dimension = undefined,
                .texture_component_type = undefined,
                .storage_texture_format = undefined,
                .count = undefined,
                .has_dynamic_offset = undefined,
                .min_buffer_binding_size = undefined,
            },
        };
        const bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(
            device,
            &(c.WGPUBindGroupLayoutDescriptor){
                .label = "bind group layout",
                .entries = &bind_group_layout_entries,
                .entries_length = bind_group_layout_entries.len,
            },
        );
        const bind_group_layouts = [_]c.WGPUBindGroup{bind_group_layout};

        ////////////////////////////////////////////////////////////////////////////
        // Render pipelines
        const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(
            device,
            &(c.WGPUPipelineLayoutDescriptor){
                .bind_group_layouts = &bind_group_layouts,
                .bind_group_layouts_length = bind_group_layouts.len,
            },
        );
        defer c.wgpuPipelineLayoutRelease(pipeline_layout);

        const render_pipeline = c.wgpuDeviceCreateRenderPipeline(
            device,
            &(c.WGPURenderPipelineDescriptor){
                .layout = pipeline_layout,
                .vertex_stage = (c.WGPUProgrammableStageDescriptor){
                    .module = vert_shader,
                    .entry_point = "main",
                },
                .fragment_stage = &(c.WGPUProgrammableStageDescriptor){
                    .module = frag_shader,
                    .entry_point = "main",
                },
                .rasterization_state = &(c.WGPURasterizationStateDescriptor){
                    .front_face = c.WGPUFrontFace_Ccw,
                    .cull_mode = c.WGPUCullMode_None,
                    .depth_bias = 0,
                    .depth_bias_slope_scale = 0.0,
                    .depth_bias_clamp = 0.0,
                },
                .primitive_topology = c.WGPUPrimitiveTopology_TriangleList,
                .color_states = &(c.WGPUColorStateDescriptor){
                    .format = c.WGPUTextureFormat_Bgra8Unorm,
                    .alpha_blend = (c.WGPUBlendDescriptor){
                        .src_factor = c.WGPUBlendFactor_One,
                        .dst_factor = c.WGPUBlendFactor_Zero,
                        .operation = c.WGPUBlendOperation_Add,
                    },
                    .color_blend = (c.WGPUBlendDescriptor){
                        .src_factor = c.WGPUBlendFactor_One,
                        .dst_factor = c.WGPUBlendFactor_Zero,
                        .operation = c.WGPUBlendOperation_Add,
                    },
                    .write_mask = c.WGPUColorWrite_ALL,
                },
                .color_states_length = 1,
                .depth_stencil_state = null,
                .vertex_state = (c.WGPUVertexStateDescriptor){
                    .index_format = c.WGPUIndexFormat_Uint16,
                    .vertex_buffers = null,
                    .vertex_buffers_length = 0,
                },
                .sample_count = 1,
                .sample_mask = 0,
                .alpha_to_coverage_enabled = false,
            },
        );

        return Self{
            .device = device,
            .tex_sampler = tex_sampler,
            .render_pipeline = render_pipeline,
            .bind_group_layout = bind_group_layout,
            .bind_group = null, // Not assigned until bind_to_tex is called
        };
    }

    pub fn bind_to_tex(self: *Self, tex_view: c.WGPUTextureView) void {
        if (self.bind_group) |b| {
            c.wgpuBindGroupRelease(b);
        }
        const bind_group_entries = [_]c.WGPUBindGroupEntry{
            (c.WGPUBindGroupEntry){
                .binding = 0,
                .texture_view = tex_view,
                .sampler = 0, // None
                .buffer = 0, // None

                .offset = undefined,
                .size = undefined,
            },
            (c.WGPUBindGroupEntry){
                .binding = 1,
                .sampler = self.tex_sampler,
                .texture_view = 0, // None
                .buffer = 0, // None

                .offset = undefined,
                .size = undefined,
            },
        };
        self.bind_group = c.wgpuDeviceCreateBindGroup(
            self.device,
            &(c.WGPUBindGroupDescriptor){
                .label = "bind group",
                .layout = self.bind_group_layout,
                .entries = &bind_group_entries,
                .entries_length = bind_group_entries.len,
            },
        );
    }

    pub fn deinit(self: *Self) void {
        c.wgpuSamplerRelease(self.tex_sampler);
        c.wgpuBindGroupLayoutRelease(self.bind_group_layout);
        if (self.bind_group) |b| {
            c.wgpuBindGroupRelease(b);
        }
    }

    pub fn redraw(
        self: *Self,
        next_texture: c.WGPUSwapChainOutput,
        cmd_encoder: c.WGPUCommandEncoder,
    ) void {
        const color_attachments = [_]c.WGPURenderPassColorAttachmentDescriptor{
            (c.WGPURenderPassColorAttachmentDescriptor){
                .attachment = next_texture.view_id,
                .resolve_target = 0,
                .channel = (c.WGPUPassChannel_Color){
                    .load_op = c.WGPULoadOp_Load,
                    .store_op = c.WGPUStoreOp_Store,
                    .clear_value = (c.WGPUColor){
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
            &(c.WGPURenderPassDescriptor){
                .color_attachments = &color_attachments,
                .color_attachments_length = color_attachments.len,
                .depth_stencil_attachment = null,
            },
        );

        c.wgpuRenderPassEncoderSetPipeline(rpass, self.render_pipeline);
        const b = self.bind_group orelse std.debug.panic(
            "Tried to blit preview before texture was bound",
            .{},
        );
        c.wgpuRenderPassEncoderSetBindGroup(rpass, 0, b, null, 0);
        c.wgpuRenderPassEncoderDraw(rpass, 6, 1, 0, 0);
    }
};
