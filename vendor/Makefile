all:
	@echo "Usage: make [wgpu]"

.PHONY: wgpu

WGPU=wgpu-macos-64-release.zip
wgpu:
	wget https://github.com/gfx-rs/wgpu-native/releases/download/v0.6.0/${WGPU}
	unzip -dwgpu ${WGPU}
	rm ${WGPU}
	cd wgpu && install_name_tool -id "@executable_path/../../vendor/wgpu/libwgpu_native.dylib" libwgpu_native.dylib
