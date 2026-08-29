# pixelforge

[![CI](https://github.com/hgaiser/pixelforge/workflows/CI/badge.svg)](https://github.com/hgaiser/pixelforge/actions)
[![Crates.io](https://img.shields.io/crates/v/pixelforge.svg)](https://crates.io/crates/pixelforge)
[![Documentation](https://docs.rs/pixelforge/badge.svg)](https://docs.rs/pixelforge)

A Vulkan-based video encoding and decoding library for Rust, supporting H.264,
H.265 and AV1 encode, and H.264 decode.

## Features

- **Hardware-accelerated** video encoding and decoding using Vulkan Video extensions.
- **Multiple codec support**: H.264/AVC, H.265/HEVC, AV1 encode; H.264 decode.
- **Asynchronous pipelines**: both directions submit without waiting and hand
  back a future ([`EncodeFuture`], [`DecodeFuture`]).
- **GPU color conversion**: RGB/BGR → YUV via Vulkan compute shaders (BT.709, BT.2020, sRGB→BT.2020+PQ, scRGB-linear→BT.2020+PQ).
- **HDR support**: 10-bit encoding (P010, YUV444P10), PQ transfer function, BT.2020 color space.
- **GPU-native API**: Encode directly from Vulkan images (`vk::Image`).
- **Flexible configuration**: Rate control (CBR, VBR, CQP), quality levels, GOP settings.
- **Multiple input formats**: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
- **Utility helpers**: [`InputImage`] for easy YUV data upload to GPU.
- **Optional DMA-BUF support**: Zero-copy image import from external processes (Linux only).

> **Note**: B-frame support is not yet implemented. Setting `b_frame_count > 0` will panic.

## Supported Codecs

| Codec | Encode | Decode |
|-------|--------|--------|
| H.264/AVC | ✓ | ✓ |
| H.265/HEVC | ✓ | |
| AV1 | ✓ | |

H.264 decoding is verified byte-identical to `ffmpeg -pix_fmt nv12` on AMD
(RADV), NVIDIA and Intel (ANV).

## Requirements

- A GPU with Vulkan video support (e.g., NVIDIA RTX series, AMD RDNA2+, Intel Arc).
  Decoding additionally needs a video decode queue; on Intel Arc under Mesa it
  currently has to be enabled with `ANV_DEBUG=video-decode,video-encode`.

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

### Decoding Video

The decoder is stream-driven: it creates its Vulkan session from the
stream's own parameter sets, so nothing has to be configured up front, and a
mid-stream resolution change is handled transparently.

[`Decoder::decode`](decoder::Decoder::decode) submits without waiting and
returns a [`DecodeFuture`], mirroring [`Encoder::encode`]. Keep a couple in
flight to overlap parsing with GPU decode. Frames come out in presentation
order without ever being copied: a picture waiting its turn stays pinned in
the DPB slot it was decoded into. Drop each frame when done, which returns
its storage to the decoder and is what keeps decoding running.

```rust
use pixelforge::{Codec, VideoContextBuilder};
use pixelforge::decoder::{DecodeConfig, Decoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .app_name("Decoder Example")
        .require_decode(Codec::H264)
        .build()?;

    let mut decoder = Decoder::new(context, DecodeConfig::h264())?;
    let stream: Vec<u8> = std::fs::read("input.264")?;

    // `split` carves the stream into one chunk per coded frame.
    for (i, unit) in decoder.split(&stream).enumerate() {
        for frame in pollster::block_on(decoder.decode(unit, i as u64)?)? {
            // `frame.image` is a decoder-owned GPU image, valid until dropped.
            let data = decoder.download(&frame)?;
            let _ = (data.y, data.uv);
        }
    }
    // Drain whatever the reorder buffer still holds.
    for frame in pollster::block_on(decoder.flush()?)? {
        let _ = decoder.download(&frame)?;
    }
    Ok(())
}
```

A held frame reserves a DPB slot, so
[`DecodeConfig::with_output_depth`](decoder::DecodeConfig::with_output_depth)
bounds how many the caller may hold before `decode` blocks waiting for one
back. On a device with too few slots for the stream, frames fall back to a
copy rather than failing.

### Color Conversion (RGB → YUV)

PixelForge includes a GPU compute shader for converting RGB input to YUV output, supporting multiple color spaces:

| Color Space | Description |
|-------------|-------------|
| `Bt709` | Standard SDR (BT.709 coefficients) |
| `Bt2020` | HDR passthrough (BT.2020 coefficients, PQ-encoded input) |
| `SrgbToBt2020Pq` | SDR-in-HDR (sRGB → linear → BT.2020 gamut → PQ OETF) |
| `Bt709LinearToBt2020Pq` | scRGB HDR (linear BT.709 → BT.2020 gamut → PQ OETF). `sdr_reference_white_nits` sets the interpretation of 1.0; per the scRGB spec (IEC 61966-2-2), 80 nits. |

Supported input formats: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
Supported output formats: NV12 (8-bit), I420 (8-bit), YUV444 (8-bit), P010 (10-bit), YUV444P10 (10-bit).

```rust
use pixelforge::{ColorConverter, ColorConverterConfig, ColorSpace, InputFormat, OutputFormat, VideoContextBuilder};

let context = VideoContextBuilder::new()
    .app_name("Color Converter")
    .build()?;

let mut config = ColorConverterConfig::new(1920, 1080, InputFormat::BGRx, OutputFormat::NV12);
config.color_space = ColorSpace::SrgbToBt2020Pq;

let mut converter = ColorConverter::new(context.clone(), config)?;
// converter.convert(input_image, output_buffer)?;
```

## Benchmarking

Run the encode latency benchmark with:

```
cargo run --example encode_bench
```

## Examples

Run the examples with:

```
# Query codec capabilities
cargo run --example query_capabilities

# H.264 decoding to raw YUV
cargo run --example decode_h264 -- input.264 output.yuv

# H.264 encoding example
cargo run --example encode_h264

# H.265 encoding example
cargo run --example encode_h265

# AV1 encoding example
cargo run --example encode_av1

# Verify all codecs and formats
cargo run --example verify_all
```

## Shader Development

The color conversion shader is precompiled to SPIR-V and embedded at build time.
See [shader/README.md](shader/README.md) for details on editing and recompiling shaders.

## TODO's

1. [] H.265 and AV1 decoding.
1. [] B-frames support (encode).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgement

This project was heavily inspired by the [vk_video_samples](https://github.com/nvpro-samples/vk_video_samples)
repository by NVIDIA, which provided invaluable reference for Vulkan Video encoding.

License: BSD-2-Clause
