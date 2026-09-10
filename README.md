# pixelforge

[![CI](https://github.com/hgaiser/pixelforge/workflows/CI/badge.svg)](https://github.com/hgaiser/pixelforge/actions)
[![Crates.io](https://img.shields.io/crates/v/pixelforge.svg)](https://crates.io/crates/pixelforge)
[![Documentation](https://docs.rs/pixelforge/badge.svg)](https://docs.rs/pixelforge)

A Vulkan-based video encoding and decoding library for Rust, supporting H.264,
H.265 and AV1 encode, and H.264 decode.

## Features

- **Hardware-accelerated** video encoding and decoding using Vulkan Video extensions.
- **Multiple codec support**: H.264/AVC, H.265/HEVC, AV1 encode; H.264 decode.
- **Asynchronous pipelines**: both directions submit without waiting.
  Encoding hands back an [`EncodeFuture`]; decoding delivers frames through a
  [`DecodeSource`] as the GPU finishes with them.
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

Bytes go in through a [`DecodeSink`], frames come out
of a [`DecodeSource`]. A [`Decoder`] holds both, so
one thread can drive the whole thing; [`Decoder::split`](decoder::Decoder::split)
separates them for a producer and a consumer on their own threads.

Frames come out in presentation order and, where the device supports unified
image layouts, without ever being copied: the frame *is* the decoder's own
image. Drop each one when done, which returns its storage.

```rust
use pixelforge::{Codec, VideoContextBuilder};
use pixelforge::decoder::{DecodeConfig, Decoder, FramePoll};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .app_name("Decoder Example")
        .require_decode(Codec::H264)
        .build()?;

    // A file can cut anywhere, so let the decoder frame it. Input that
    // arrives already framed (RTP, a container) skips `with_byte_stream`.
    let config = DecodeConfig::h264().with_byte_stream();
    let mut decoder = Decoder::new(context, config)?;
    let stream: Vec<u8> = std::fs::read("input.264")?;

    for (i, chunk) in stream.chunks(64 * 1024).enumerate() {
        // The status says what happened; an `Err` means something is
        // actually wrong. Joining a stream partway through is not.
        let _status = decoder.decode(chunk, i as u64)?;
        // Take what the GPU has finished with; `Pending` just means "not yet".
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            // `frame.image` is a decoder-owned GPU image, valid until dropped.
            let _ = frame.image;
        }
    }

    // End of stream: decodes the trailing picture, emits what reordering
    // held back, and closes the source.
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        let _ = frame.image;
    }
    Ok(())
}
```

A live frame reserves a DPB slot, so
[`DecodeConfig::with_output_depth`](decoder::DecodeConfig::with_output_depth)
bounds how many can be outstanding before the decoder starts copying
pictures out instead of handing over its own. Reading a frame back to the
CPU is the consumer's job; `examples/common` shows one way.

### Color Conversion (RGB → YUV)

PixelForge includes a GPU compute shader for converting RGB input to YUV
output. The conversion is two decisions: what the input already is, and
what the encoded stream should be.

| `SourceColor` | What the input pixels are |
|---------------|---------------------------|
| `Srgb` | BT.709 primaries, sRGB transfer. Ordinary SDR content. |
| `Bt709Linear` | Linear BT.709. This is scRGB, from an `EXTENDED_SRGB_LINEAR_EXT` swapchain. |
| `Bt2020Linear` | Linear BT.2020. |
| `Bt2020Pq` | BT.2020 primaries, already PQ-encoded. |

| `TargetColor` | What the stream is |
|---------------|--------------------|
| `Bt709` | BT.709 primaries, transfer and matrix. SDR. |
| `Bt2020Pq` | BT.2020 primaries, PQ transfer, BT.2020 NCL matrix. HDR10. |

The relative sources carry the luminance that a sample value of 1.0 stands
for, since that is a property of the source and not of the target: 203 nits
for `Srgb` per ITU-R BT.2408, and 80 for scRGB per IEC 61966-2-2. It is read
only on the way to `Bt2020Pq`, where the PQ encode needs it to be absolute.

Every source reaches `Bt2020Pq`. Only `Srgb` reaches `Bt709`, because the
others would need a forward gamma encode or tone mapping; those pairings are
rejected by [`ColorConverter::new`] rather than quietly passed through.

Supported input formats: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
Supported output formats: NV12 (8-bit), I420 (8-bit), YUV444 (8-bit), P010 (10-bit), YUV444P10 (10-bit).

Because the target and the range fully determine the stream's colour
signalling, [`ColorConverterConfig::color_description`] derives the
encoder's VUI declaration from the conversion itself. Use it rather than
declaring the same thing twice: full-range samples tagged as limited are
expanded a second time on playback, and the API cannot catch that if the two
are set independently.

```rust
use pixelforge::{
    Codec, ColorConverter, ColorConverterConfig, ColorRange, EncodeConfig, Encoder,
    InputFormat, OutputFormat, SourceColor, TargetColor, VideoContextBuilder,
};

let context = VideoContextBuilder::new()
    .app_name("Color Converter")
    .require_encode(Codec::H265)
    .build()?;

// SDR desktop content, encoded as HDR10 for an HDR streaming session.
let config = ColorConverterConfig::new(1920, 1080, InputFormat::BGRx, OutputFormat::P010)
    .with_source(SourceColor::srgb())
    .with_target(TargetColor::Bt2020Pq)
    .with_range(ColorRange::Full);

// The encoder declares exactly what the shader wrote.
let encode_config = EncodeConfig::h265(1920, 1080)
    .with_color_description(config.color_description());

let mut converter = ColorConverter::new(context.clone(), config)?;
let mut encoder = Encoder::new(context, encode_config)?;
// converter.convert(input_image, layout, encoder.input_image())?;
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
