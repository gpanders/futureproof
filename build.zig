const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "futureproof",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Libraries!
    exe.linkSystemLibrary("glfw3");
    exe.linkSystemLibrary("freetype2");
    exe.linkSystemLibrary("stdc++"); // needed for shaderc

    exe.addLibraryPath(b.path("vendor/wgpu"));
    exe.linkSystemLibrary("wgpu_native");
    exe.addIncludePath(b.path("vendor")); // "wgpu/wgpu.h" is the wgpu header

    exe.addLibraryPath(b.path("vendor/shaderc/lib"));
    exe.linkSystemLibrary("shaderc_combined");
    exe.addIncludePath(b.path("vendor/shaderc/include/"));

    exe.addIncludePath(b.path(".")); // for "extern/futureproof.h"

    // This must come before the install_name_tool call below
    b.installArtifact(exe);

    if (target.result.isDarwin()) {
        // exe.addFrameworkPath(b.path("/System/Library/Frameworks"));
        exe.linkFramework("Foundation");
        exe.linkFramework("AppKit");
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
