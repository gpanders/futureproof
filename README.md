## About
*Futureproof* is a live editor for GPU shaders, built on Zig, Neovim, and WebGPU.

![Seascape](https://www.mattkeeter.com/projects/futureproof/seascape@2x.png)

[Project homepage](https://mattkeeter.com/projects/futureproof)

## Building

### macOS (x86)

Install `freetype`, `glfw3`, and `shaderc` through [Homebrew](https://brew.sh):
```
brew install freetype glfw3 shaderc
```

Get vendored dependencies:
```
cd futureproof/vendor
make wgpu
```

Build using Zig, using a recent [nightly build](https://ziglang.org/download/) (0.7.1, after 2020-12-31)
```
cd futureproof
zig build run
```

(You may need `env ZIG_SYSTEM_LINKER_HACK=1`, depending on Zig compiler version)

### Other OS
Good luck - open a PR if you get things working!

## Project status
![Project done](https://img.shields.io/badge/status-done-blue.svg) ![Project unmaintained](https://img.shields.io/badge/project-unmaintained-red.svg)

This project is **done**, and I don't plan to maintain it in the future.

It is only claimed to work on my laptop,
and even then,
will probably break if the Zig compiler version changes.

I'm unlikely to fix any issues,
although I will optimistically merge small-to-medium PRs that fix bugs
or add support for more platforms.

If you'd like to add major features, please fork the project;
I'd be happy to link to any forks which achieve critical momentum!

## License

Licensed under either of

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
