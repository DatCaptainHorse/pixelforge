# Agent Instructions

## Project Overview
Pixelforge is a Rust library for video encoding and decoding using Vulkan Video.

## Build & Test
```bash
cargo build
cargo test
cargo run --example encode_h264
cargo run --example decode_h264 -- input.264 output.yuv
```

Decoding on Intel Arc under Mesa needs the video queues enabled explicitly:
`ANV_DEBUG=video-decode,video-encode`.

## README Generation
The `README.md` is generated from the doc comments in `src/lib.rs` using the `README.tpl` template.
To regenerate:
```bash
cargo readme > README.md
```

This is the exact command CI compares against (`diff --brief <(cargo readme)
README.md`). Do not add `--no-indent-headings`: it flattens every section to `#`
and the diff then fails.
Do not edit `README.md` directly; update the doc comments in `src/lib.rs` instead.

To verify the quality of the encoded videos, run:

```bash
cargo run --example encode_h265 \
    && rm -f decoded.yuv \
    && ffmpeg -hide_banner -loglevel error -y -i output.h265 -pix_fmt yuv420p -f rawvideo decoded.yuv \
    && ffmpeg -hide_banner -loglevel info -s 320x240 -pix_fmt yuv420p -f rawvideo -i testdata/test_frames.yuv -s 320x240 -pix_fmt yuv420p -f rawvideo -i decoded.yuv -lavfi psnr -f null -
```

To verify decoding, decode a stream and compare against ffmpeg's software
decoder, which should be byte-identical:

```bash
cargo run --example decode_h264 -- tests/data/bframes.264 out.yuv \
    && ffmpeg -hide_banner -loglevel error -y -i tests/data/bframes.264 -pix_fmt nv12 ref.yuv \
    && cmp ref.yuv out.yuv && echo "decode matches ffmpeg"
```

Decoding has two paths and both need checking. Where the device supports
`VK_KHR_unified_image_layouts`, frames are the decoder's own images, handed over
with no copy. Where it does not, they are copied into private images. Force the
copying path to exercise it on hardware that would otherwise never take it:

```bash
PIXELFORGE_NO_UNIFIED_LAYOUTS=1 cargo run --example decode_h264 -- \
    tests/data/bframes.264 out.yuv && cmp ref.yuv out.yuv
```

Which path a run took is in the debug log (`RUST_LOG=debug`), on the
`H.264 decode session:` line as `pinnable=`.

The render path has its own example. `sample_frame` reads decoded frames in a
compute shader through a sampler-YCbCr conversion, which is the path that never
copies anything, and writes RGBA:

```bash
cargo run --example sample_frame -- tests/data/bframes.264 out.rgba
```

Do not judge that one by PSNR. The hardware sampler and ffmpeg reconstruct
subsampled chroma differently, so they agree almost everywhere and disagree
hard on pixels sitting on a colour edge; on the synthetic test patterns that
drags PSNR to ~26 dB while the images are indistinguishable. Compare the share
of pixels that agree instead: both AMD and Intel land on 87.9% within 2 and
33.2% exact for `bframes.264`.

`sample_planes` is the other half of that story, and the tighter test. It reads
the same frames through per-plane views, with no ycbcr conversion and no sampler
at all, and writes the samples straight through, so its output is plain NV12 and
can be compared exactly:

```bash
cargo run --example sample_planes -- tests/data/bframes.264 out.yuv \
    && cmp ref.yuv out.yuv && echo "plane views match the decoder"
```

Run that one on AMD too, not just locally. Its frames are pool copies on a
device without unified image layouts, which carry no video usage, so a view
mistake that only a real DPB image would catch passes silently on Intel.

Make sure there are no Vulkan validation layer errors during execution. Enable
them with `PIXELFORGE_VALIDATION=1`; the layer's messages are routed through
`tracing`, so pair it with `RUST_LOG=warn` (or `debug` for the layer's own
chatter). Without `VK_LAYER_KHRONOS_validation` installed, pixelforge logs a
warning and carries on with validation disabled, so absence of errors means
nothing if the layer is missing.

## Code Style
- Follow `rustfmt.toml` formatting rules
- Run `cargo fmt` before committing
- Use clippy for linting
- If a comment is a sentence, it should end with a period
- Avoid long files; split into modules if necessary

## Project Structure
- `src/` - Library source code
- `examples/` - Usage examples for encoding and decoding
- `testdata/` - Test input files (note: `test_frames.yuv` is a git-LFS pointer;
  generate frames locally if LFS content is unavailable)
- `tests/data/` - Small H.264 streams used by the decoder tests

## Key Dependencies
- Vulkan Video API
- ash (Vulkan bindings for Rust)
