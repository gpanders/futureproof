WGPU = wgpu-macos-aarch64-release.zip
WGPU_VERSION = 0.19.4.1

all:
	@echo "Usage: make [wgpu]"


.PHONY: wgpu
wgpu:
	wget https://github.com/gfx-rs/wgpu-native/releases/download/v${WGPU_VERSION}/${WGPU}
	unzip -dwgpu ${WGPU}
	rm ${WGPU}

.PHONY: clean
clean:
	$(RM) -r wgpu
