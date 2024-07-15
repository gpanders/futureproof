const std = @import("std");
const builtin = @import("builtin");

// Returns the file contents, loaded from the file in debug builds and
// compiled in with release builds.  alloc must be an arena allocator,
// because otherwise there will be a leak.
pub fn file_contents(arena: *std.heap.ArenaAllocator, comptime name: []const u8) ![]const u8 {
    switch (builtin.mode) {
        .Debug => {
            const file = try std.fs.cwd().openFile(name, .{});
            return try file.reader().readAllAlloc(arena.allocator(), std.math.maxInt(u32));
        },
        else => {
            const f = @embedFile("../" ++ name);
            return f[0..];
        },
    }
}
