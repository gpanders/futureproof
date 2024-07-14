const builtin = @import("builtin");
const std = @import("std");

const c = @import("c.zig");

pub fn get_clipboard() [*c]u8 {
    const platform = builtin.os.tag;
    if (platform == .macos) {
        const objc = @import("objc.zig");
        const darwin = @import("darwin.zig");

        const pasteboard_class = objc.class("NSPasteboard");
        const pb = objc.call(pasteboard_class, "generalPasteboard");
        const items = objc.call(pb, "pasteboardItems");

        const item = objc.call_(items, "objectAtIndex:", @as(c_ulong, 0));
        const str = objc.call_(item, "stringForType:", darwin.NSPasteboardTypeString);

        return @ptrCast(objc.call(str, "UTF8String"));
    } else {
        std.debug.panic("Unimplemented on platform {}", .{platform});
    }
}
