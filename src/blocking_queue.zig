const std = @import("std");

pub fn BlockingQueue(comptime T: type, comptime capacity: usize) type {
    return struct {
        items: [capacity]T = undefined,
        read: usize = 0,
        write: usize = 0,
        empty: bool = true,
        lock: std.Thread.Mutex = .{},
        cv: std.Thread.Condition = .{},

        const Self = @This();

        pub fn put(self: *Self, value: T) !void {
            self.lock.lock();
            defer self.lock.unlock();

            if (self.read == self.write and !self.empty) {
                return error.OutOfMemory;
            }

            self.items[self.write] = value;
            self.write = self.write + 1;
            if (self.write >= capacity) {
                self.write -= capacity;
            }

            self.empty = false;
            self.cv.signal();
        }

        pub fn get(self: *Self) T {
            self.lock.lock();
            defer self.lock.unlock();
            while (self.empty) {
                self.cv.wait(&self.lock);
            }

            const value = self.tryGet() orelse
                // We already waited for the queue to be non-empty
                unreachable;

            return value;
        }

        pub fn tryGet(self: *Self) ?T {
            self.lock.lock();
            defer self.lock.unlock();

            if (self.empty) {
                return null;
            }

            const value = self.items[self.read];
            self.read = self.read + 1;
            if (self.read >= capacity) {
                self.read -= capacity;
            }

            self.empty = self.read == self.write;
            return value;
        }
    };
}
