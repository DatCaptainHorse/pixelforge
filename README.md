[![CI](https://github.com/hgaiser/pixelforge/workflows/CI/badge.svg)](https://github.com/hgaiser/pixelforge/actions)
[![Crates.io](https://img.shields.io/crates/v/pixelforge.svg)](https://crates.io/crates/pixelforge)
[![Documentation](https://docs.rs/pixelforge/badge.svg)](https://docs.rs/pixelforge)

# PixelForge

A Vulkan-based video encoding library for Rust, supporting H.264, H.265, and AV1 codecs.

> ⚠️ **Disclaimer**: This library was developed using AI ("vibe-coding") - partly to
> see if it could be done, partly because I have practically zero experience with Vulkan.
> While the code has been tested and works in my usecase, it may not work in all cases
> and it may not follow best practices. Contributions and improvements are very welcome!

## Features

- **Hardware-accelerated** video encoding using Vulkan Video extensions.
- **Multiple codec support**: H.264/AVC, H.265/HEVC, AV1.
- **GPU-native API**: Encode directly from Vulkan images (`vk::Image`).
- **Flexible configuration**: Rate control (CBR, VBR, CQP), quality levels, GOP settings.
- **Utility helpers**: [`InputImage`] for easy YUV data upload to GPU.
- **Optional DMA-BUF support**: Zero-copy image import from external processes (Linux only).

> **Note**: B-frame support is not yet implemented. Setting `b_frame_count > 0` will panic.

## Supported Codecs

| Codec | Encode |
|-------|--------|
| H.264/AVC | ✓ |
| H.265/HEVC | ✓ |
| AV1 | ✓ (experimental) |

> ⚠️ **AV1 Warning**: AV1 encoding is experimental. On NVIDIA GPUs, P-frames cannot
> reference other P-frames, causing all P-frames to reference the I-frame instead. This
> leads to progressively larger frame sizes over time. Consider using H.264 or HEVC
> until this is resolved.

## Requirements

- A GPU with Vulkan video encoding support (e.g., NVIDIA RTX series, AMD RDNA2+, Intel Arc)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
pixelforge = "0.1"
```

### Optional Features

| Feature | Description |
|---------|-------------|
| `dmabuf` | Enable DMA-BUF support for zero-copy image import from external processes (Linux only). Adds Vulkan extensions: `VK_KHR_external_memory`, `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`. |

To enable DMA-BUF support:

```toml
[dependencies]
pixelforge = { version = "0.1", features = ["dmabuf"] }
```

## Quick Start

### Query Capabilities

```rust
use pixelforge::{Codec, VideoContextBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .app_name("My App")
        .build()?;

    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        println!("{:?}: encode={}",
            codec,
            context.supports_encode(codec)
        );
    }
    Ok(())
}
```

### Encoding Video

```rust
use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .app_name("Encoder Example")
        .require_encode(Codec::H264)
        .build()?;

    let config = EncodeConfig::h264(1920, 1080)
        .with_rate_control(RateControlMode::Vbr)
        .with_target_bitrate(5_000_000)
        .with_frame_rate(30, 1)
        .with_gop_size(60);

    // Create an InputImage helper for uploading YUV data to the GPU.
    let mut input_image = InputImage::new(
        context.clone(),
        Codec::H264,
        1920,
        1080,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;
    let mut encoder = Encoder::new(context, config)?;

    // For each frame: upload YUV data and encode.
    // let yuv_data: &[u8] = ...;  // YUV420 frame data
    // input_image.upload_yuv420(yuv_data)?;
    // let packets = encoder.encode(input_image.image())?;

    Ok(())
}
```

## Examples

Run the examples with:

```
# Query codec capabilities
cargo run --example query_capabilities

# H.264 encoding example
cargo run --example encode_h264

# H.265 encoding example
cargo run --example encode_h265

# AV1 encoding example
cargo run --example encode_av1
```

## TODO's

1. [] Decoding.
1. [] B-frames support.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

License: BSD-2-Clause
