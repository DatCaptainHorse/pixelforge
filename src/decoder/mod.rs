//! Hardware-accelerated video decoding using Vulkan Video.
//!
//! The decoder consumes a raw byte stream (Annex B for H.264: start-code
//! delimited NAL units) and produces decoded frames as GPU images
//! (`vk::Image`), optionally with CPU readback.
//!
//! # Design
//!
//! - **Stream-driven**: the Vulkan video session is created lazily from the
//!   stream's own parameter sets, so no dimensions or profile need to be
//!   configured up front. Mid-stream resolution changes recreate the session
//!   transparently.
//! - **Display order by default**: frames come out in presentation order (see
//!   [`OutputOrder`]); drain the reorder buffer with [`Decoder::flush`] at end
//!   of stream. Streams without B-frames add no latency. Switch to
//!   [`OutputOrder::Decode`] for the lowest-latency decode-order output.
//! - **Zero-copy output**: [`DecodedFrame::image`] is a decoder-owned GPU
//!   image. See [`DecodedFrame`] for validity rules.
//!
//! # Limitations (H.264)
//!
//! - Progressive streams only (no interlaced/field coding).
//! - Streams using explicit scaling matrices are rejected.
//!
//! Reference management is complete (IDR, sliding-window, and MMCO), so any
//! stream this crate's encoder produces — and typical output from x264, NVENC
//! and friends, including B-pyramid — decodes correctly.

pub(crate) mod bitreader;
pub(crate) mod codec;
/// H.264 decoding.
pub mod h264;

use crate::encoder::{BitDepth, Codec, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;

/// The order in which [`Decoder::decode`] returns frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputOrder {
    /// Presentation order (default). Frames come out sorted by picture order
    /// count, matching how a player would show them. The decoder buffers a
    /// bounded number of frames (the stream's `max_num_reorder_frames`), so
    /// call [`Decoder::flush`] at end of stream to drain the rest. For streams
    /// without B-frames this adds no latency and behaves like [`Self::Decode`].
    Display,
    /// Decode order: frames are returned the moment their GPU work completes,
    /// the lowest-latency option. With B-frames the caller must reorder by
    /// [`DecodedFrame::poc`] if presentation order matters.
    Decode,
}

/// Configuration for creating a [`Decoder`].
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// The codec to decode.
    pub codec: Codec,
    /// The order frames are returned in. Defaults to [`OutputOrder::Display`].
    pub output_order: OutputOrder,
}

impl DecodeConfig {
    /// Create an H.264 decode configuration (display-order output).
    pub fn h264() -> Self {
        Self {
            codec: Codec::H264,
            output_order: OutputOrder::Display,
        }
    }

    /// Return frames in decode order for lowest latency, rather than sorting
    /// them into presentation order. See [`OutputOrder`].
    pub fn with_decode_order(mut self) -> Self {
        self.output_order = OutputOrder::Decode;
        self
    }
}

/// A decoded frame residing in GPU memory.
///
/// # Validity
///
/// [`image`](Self::image) is a decoder-owned GPU image. It remains valid and
/// unmodified until **the next call to [`Decoder::decode`] or
/// [`Decoder::flush`]**, at which point the decoder may reuse it. For zero-copy
/// consumption, use the image (sample it, copy it, encode from it) before
/// feeding more data to the decoder; for retention, use [`Decoder::download`]
/// or copy it to an image you own.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// The decoded picture on the GPU.
    ///
    /// The image format is the decoder's picture format (see
    /// [`Decoder::picture_format`]), typically NV12
    /// (`G8_B8R8_2PLANE_420_UNORM`) or P010 for 10-bit streams. The image is
    /// in `VIDEO_DECODE_DPB_KHR` or `VIDEO_DECODE_DST_KHR` layout (see
    /// [`layout`](Self::layout)) and sized to the coded dimensions; only the
    /// `width` x `height` top-left region contains the visible picture.
    pub image: vk::Image,
    /// An image view over the whole decoded picture.
    pub image_view: vk::ImageView,
    /// Current layout of `image`.
    pub layout: vk::ImageLayout,
    /// Array layer within `image` holding this picture. Always 0 for a
    /// non-layered DPB or a distinct output image; the DPB slot's layer
    /// when the DPB is layered and the driver decoded straight into it.
    pub array_layer: u32,
    /// The pixel format of this frame.
    pub pixel_format: PixelFormat,
    /// Visible (cropped) width in pixels.
    pub width: u32,
    /// Visible (cropped) height in pixels.
    pub height: u32,
    /// Coded width in pixels (image width).
    pub coded_width: u32,
    /// Coded height in pixels (image height).
    pub coded_height: u32,
    /// Presentation timestamp, passed through from [`Decoder::decode`].
    pub pts: u64,
    /// Picture order count (presentation order within a coded video sequence).
    /// In [`OutputOrder::Decode`] mode, sort by this for presentation order.
    pub poc: i32,
    /// Whether this frame is an IDR (stream random-access point).
    pub is_idr: bool,
}

/// Decoded frame data downloaded to the CPU.
#[derive(Debug, Clone)]
pub struct DecodedFrameData {
    /// Luma plane, `y_stride * height` bytes.
    pub y: Vec<u8>,
    /// Interleaved chroma plane (semi-planar: UV for NV12/P010).
    pub uv: Vec<u8>,
    /// Row stride of the luma plane in bytes.
    pub y_stride: usize,
    /// Row stride of the chroma plane in bytes.
    pub uv_stride: usize,
    /// Visible width in pixels.
    pub width: u32,
    /// Visible height in pixels.
    pub height: u32,
    /// Bit depth of the samples (8 => NV12, 10 => P010).
    pub bit_depth: BitDepth,
    /// Chroma subsampling.
    pub pixel_format: PixelFormat,
}

/// Split an Annex B stream into access units (one coded picture each).
///
/// [`Decoder::decode`] treats the end of its input as a picture boundary, so
/// callers feeding a whole file must split it first. Feeding one access unit
/// per call is also the lowest-latency usage.
///
/// A new access unit begins at each VCL NAL whose `first_mb_in_slice` is 0.
/// Parameter sets, SEI and access unit delimiters are attached to the picture
/// that follows them.
///
/// ```no_run
/// # use pixelforge::decoder::{access_units, Decoder};
/// # fn run(decoder: &mut Decoder, stream: &[u8]) -> pixelforge::error::Result<()> {
/// for (i, au) in access_units(stream).enumerate() {
///     for frame in decoder.decode(au, i as u64)? {
///         // ... use frame before the next decode() call ...
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub fn access_units(stream: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut starts: Vec<usize> = Vec::new();
    let mut pending_start: Option<usize> = None;

    for nal in h264::parser::iter_nal_units_with_offsets(stream) {
        let (offset, nal) = nal;
        if nal.nal_type.is_slice() {
            // first_mb_in_slice is the leading ue(v) of the slice header; it is
            // zero — encoded as a leading `1` bit — exactly at a picture start.
            let first_mb_is_zero = nal.payload().first().is_some_and(|b| b & 0x80 != 0);
            if first_mb_is_zero {
                starts.push(pending_start.take().unwrap_or(offset));
            }
            pending_start = None;
        } else if pending_start.is_none() {
            // Parameter sets / SEI / AUD belong to the picture they precede.
            pending_start = Some(offset);
        }
    }

    let ends: Vec<usize> = starts
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(stream.len()))
        .collect();

    starts
        .into_iter()
        .zip(ends)
        .map(move |(start, end)| &stream[start..end])
        .collect::<Vec<_>>()
        .into_iter()
}

/// The codec-erased operations every codec decoder exposes.
trait DecoderApi: Send {
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<Vec<DecodedFrame>>;
    fn flush(&mut self) -> Result<Vec<DecodedFrame>>;
    fn download(&mut self, frame: &DecodedFrame) -> Result<DecodedFrameData>;
    fn copy_frame_to_planes(
        &mut self,
        frame: &DecodedFrame,
        y_image: vk::Image,
        uv_image: vk::Image,
    ) -> Result<()>;
    fn picture_format(&self) -> Option<vk::Format>;
}

/// Video decoder supporting multiple codecs.
///
/// Constructed via [`Decoder::new`], which selects the codec from the config.
/// Mirrors [`Encoder`](crate::encoder::Encoder): all codec decoders share the
/// same generic driving flow and are held behind a single boxed pointer.
pub struct Decoder(Box<dyn DecoderApi>);

impl Decoder {
    /// Create a new decoder for the codec named in `config`.
    ///
    /// Requires a [`VideoContext`] whose device has a video decode queue and
    /// supports the codec (see
    /// [`VideoContextBuilder::require_decode`](crate::vulkan::VideoContextBuilder::require_decode)
    /// and [`VideoContext::supports_decode`]).
    pub fn new(context: VideoContext, config: DecodeConfig) -> Result<Self> {
        if !context.supports_decode(config.codec) {
            return Err(PixelForgeError::CodecNotSupported(format!(
                "{:?} decode is not supported by the selected device",
                config.codec
            )));
        }
        let display_order = config.output_order == OutputOrder::Display;
        let inner: Box<dyn DecoderApi> = match config.codec {
            Codec::H264 => Box::new(h264::H264Decoder::create(context, display_order)?),
            other => {
                return Err(PixelForgeError::CodecNotSupported(format!(
                    "{:?} decoding is not implemented yet",
                    other
                )));
            }
        };
        Ok(Decoder(inner))
    }

    /// Decode a chunk of the raw byte stream.
    ///
    /// `data` may contain any number of *complete* access units (Annex B
    /// start codes for H.264): parameter sets, one access unit, or several.
    /// The end of `data` is treated as an access-unit boundary, so a picture
    /// must not be split across calls. For lowest latency, feed exactly one
    /// access unit per call; the picture is decoded immediately and returned
    /// from the same call.
    ///
    /// `pts` is attached to every frame produced by this call.
    ///
    /// Returns decoded frames in the configured [`OutputOrder`] (display order
    /// by default). In display order the count returned per call varies — a
    /// picture may be held back until later ones are decoded — so drain the
    /// remainder with [`flush`](Self::flush) at end of stream. In decode order
    /// it is usually zero or one frame per access unit fed.
    pub fn decode(&mut self, data: &[u8], pts: u64) -> Result<Vec<DecodedFrame>> {
        self.0.decode(data, pts)
    }

    /// Return any frames still held for display-order reordering.
    ///
    /// Call once after the last [`decode`](Self::decode) to emit the tail of
    /// the stream. Frames come back in presentation order. Harmless (returns
    /// empty) in decode-order mode.
    pub fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        self.0.flush()
    }

    /// Download a decoded frame to the CPU as semi-planar YUV (NV12 / P010).
    pub fn download(&mut self, frame: &DecodedFrame) -> Result<DecodedFrameData> {
        self.0.download(frame)
    }

    /// Copy a decoded frame's planes into two caller-owned GPU images: luma into
    /// `y_image`, interleaved chroma into `uv_image`.
    ///
    /// Zero-copy handoff for a renderer sharing this decoder's device: the
    /// images survive past the next [`decode`](Self::decode) (unlike
    /// [`DecodedFrame::image`]), and are ready to sample as two separate
    /// textures. Both must live on the decoder's device and be sized to the
    /// frame's coded dimensions (`uv_image` at half size for 4:2:0). After the
    /// call both are in `TRANSFER_DST_OPTIMAL`.
    pub fn copy_frame_to_planes(
        &mut self,
        frame: &DecodedFrame,
        y_image: vk::Image,
        uv_image: vk::Image,
    ) -> Result<()> {
        self.0.copy_frame_to_planes(frame, y_image, uv_image)
    }

    /// The Vulkan format of decoded picture images, once known.
    ///
    /// `None` until the first parameter set has been consumed (the format is
    /// negotiated from the stream's profile).
    pub fn picture_format(&self) -> Option<vk::Format> {
        self.0.picture_format()
    }
}
