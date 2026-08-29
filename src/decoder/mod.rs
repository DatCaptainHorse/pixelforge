//! Hardware-accelerated video decoding using Vulkan Video.
//!
//! The decoder consumes a coded byte stream and produces decoded frames as GPU
//! images (`vk::Image`), optionally with CPU readback. Each [`Decoder::decode`]
//! call takes the bytes of one coded frame, which [`Decoder::split`] carves out
//! of a larger buffer using whatever framing the codec uses (Annex B start
//! codes for H.264/H.265, OBU temporal units for AV1).
//!
//! # Design
//!
//! - **Asynchronous**: [`Decoder::decode`] records and submits, then returns a
//!   [`DecodeFuture`] rather than waiting for the GPU, so the CPU can parse and
//!   submit the next picture while the current one decodes. Mirrors
//!   [`Encoder::encode`](crate::encoder::Encoder::encode).
//! - **Stream-driven**: the Vulkan video session is created lazily from the
//!   stream's own parameter sets, so no dimensions or profile need to be
//!   configured up front. Mid-stream resolution changes recreate the session
//!   transparently.
//! - **Display order**: frames come out in presentation order; drain the
//!   reorder buffer with [`Decoder::flush`] at end of stream. Streams without
//!   B-frames buffer nothing and add no latency.
//! - **Zero-copy output**: a [`DecodedFrame`] is the decoder's own DPB image,
//!   pinned until the frame is dropped. Reordering holds pins rather than
//!   copying pictures out, so display order costs no copy either. See
//!   [`DecodedFrame`] for validity rules and what holding one costs.
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
/// The Vulkan video session and the per-decoder state every codec shares.
pub(crate) mod common;
/// Frame ownership: pool images, DPB slot pins, display-order reordering.
pub(crate) mod frames;
/// H.264 decoding.
pub mod h264;
pub(crate) mod pipeline;
/// Readback and copies, all of which run on the transfer queue.
pub(crate) mod transfer;

pub use pipeline::DecodeFuture;

use crate::encoder::{BitDepth, Codec, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;

/// How many decoded frames the caller may hold at once, by default.
///
/// Matches the encoder's pipeline depth: enough to overlap consuming one frame
/// with decoding the next, without growing the decoded picture buffer further
/// than the hardware is comfortable with.
pub const DEFAULT_OUTPUT_DEPTH: usize = 2;

/// Configuration for creating a [`Decoder`].
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// The codec to decode.
    pub codec: Codec,
    /// How many decoded frames the caller may hold at once. Defaults to
    /// [`DEFAULT_OUTPUT_DEPTH`]. See [`DecodeConfig::with_output_depth`].
    pub output_depth: usize,
}

impl DecodeConfig {
    /// Create an H.264 decode configuration.
    pub fn h264() -> Self {
        Self {
            codec: Codec::H264,
            output_depth: DEFAULT_OUTPUT_DEPTH,
        }
    }

    /// How many decoded frames the caller may hold at once.
    ///
    /// A frame is the decoder's own DPB image, so holding one keeps a DPB slot
    /// reserved. The decoder allocates this many slots beyond what the stream's
    /// references and reorder depth need, and [`Decoder::decode`] blocks once
    /// every one of them is held, until a frame is dropped. Raise it to let a
    /// consumer fall further behind, at the cost of one decoded picture's worth
    /// of memory per slot.
    ///
    /// Clamped to what the device's DPB slot limit allows for the stream. If
    /// nothing is left over, frames fall back to a copy rather than failing.
    ///
    /// This also bounds how many [`DecodeFuture`]s can usefully be kept in
    /// flight: an unresolved future holds the frames it will deliver, and each
    /// of those holds a slot. Keeping more batches pending than there are
    /// output slots makes [`Decoder::decode`] wait for a frame only the caller
    /// can release, so keep the two numbers in step.
    pub fn with_output_depth(mut self, depth: usize) -> Self {
        self.output_depth = depth;
        self
    }
}

/// A decoded frame residing in GPU memory.
///
/// # Validity
///
/// [`image`](Self::image) is a decoder-owned GPU image whose storage this frame
/// keeps reserved. It stays valid and unmodified for as long as the frame is
/// alive; dropping the frame hands the storage back to the decoder.
///
/// Two consequences worth planning for:
///
/// - Holding frames costs the decoder something. The image *is* a DPB slot (no
///   copy), so a held frame reserves one of the
///   [`DecodeConfig::with_output_depth`] spare slots, and [`Decoder::decode`]
///   blocks once they are all held. Drop frames promptly: that is what keeps
///   the decoder running, and it is the pipeline's back-pressure.
/// - Drop every frame before the [`Decoder`], and before feeding a stream that
///   changes resolution (which rebuilds the session and its images). The
///   decoder warns rather than leaving handles silently dangling.
#[derive(Debug)]
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
    /// Position of this frame in presentation order within the current coded
    /// video sequence: the codec's own ordering value (H.26x picture order
    /// count, AV1 order hint). Only the relative ordering carries meaning, and
    /// it restarts at every keyframe. Frames are already emitted sorted by it.
    pub display_order: i32,
    /// Whether this frame is a keyframe: a random-access point starting a new
    /// coded video sequence (H.264 IDR, H.265 IRAP, AV1 key frame).
    pub is_keyframe: bool,
    /// Keeps this frame's storage reserved; releases it on drop. `None` for a
    /// frame that borrows the decoder's DPB image (see the validity rules).
    // Held purely for its `Drop`, which is what returns the storage.
    #[allow(dead_code)]
    pub(crate) pin: Option<frames::FramePin>,
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

/// The codec-erased operations every codec decoder exposes.
trait DecoderApi: Send {
    fn split_stream<'a>(&self, stream: &'a [u8]) -> Vec<&'a [u8]>;
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeFuture>;
    fn flush(&mut self) -> Result<DecodeFuture>;
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
        let inner: Box<dyn DecoderApi> = match config.codec {
            Codec::H264 => Box::new(h264::H264Decoder::create(context, &config)?),
            other => {
                return Err(PixelForgeError::CodecNotSupported(format!(
                    "{:?} decoding is not implemented yet",
                    other
                )));
            }
        };
        Ok(Decoder(inner))
    }

    /// Split a coded byte stream into one slice per coded frame.
    ///
    /// [`decode`](Self::decode) treats the end of its input as a frame
    /// boundary, so a caller holding a whole file or a large buffer must split
    /// it first. Framing is the codec's own: Annex B start codes and slice
    /// headers for H.264/H.265, OBU temporal units for AV1. Non-picture data
    /// (parameter sets, SEI, sequence headers) is attached to the coded frame
    /// that follows it, so feeding the pieces back in order is lossless.
    ///
    /// The returned slices borrow `stream`. Feeding one per [`decode`](Self::decode)
    /// call is also the lowest-latency usage.
    ///
    /// ```no_run
    /// # use pixelforge::decoder::Decoder;
    /// # fn run(decoder: &mut Decoder, stream: &[u8]) -> pixelforge::error::Result<()> {
    /// for (i, unit) in decoder.split(stream).enumerate() {
    ///     let batch = decoder.decode(unit, i as u64)?;
    ///     for frame in pollster::block_on(batch)? {
    ///         // ... use the frame, then drop it to release its storage ...
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    //
    // `use<'a>` keeps the iterator from capturing the borrow of `self`, so the
    // caller can call `decode` while iterating the split units.
    pub fn split<'a>(&self, stream: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + use<'a> {
        self.0.split_stream(stream).into_iter()
    }

    /// Decode a chunk of the coded byte stream.
    ///
    /// `data` may hold any number of *complete* coded frames, plus whatever
    /// non-picture data the codec carries alongside them (H.26x parameter sets
    /// and SEI, AV1 sequence headers). The end of `data` is treated as a frame
    /// boundary, so a picture must not be split across calls: use
    /// [`split`](Self::split) to carve a buffer into decodable pieces. For
    /// lowest latency feed exactly one coded frame per call; the picture is
    /// decoded immediately and returned from the same call.
    ///
    /// `pts` is attached to every frame produced by this call.
    ///
    /// Returns immediately with a [`DecodeFuture`], without waiting for the GPU:
    /// the picture is recorded and submitted, and awaiting the future yields the
    /// frames that call emits once their decode has completed. Keep a couple of
    /// futures in flight to overlap parsing and submission with GPU decode, the
    /// same way [`Encoder::encode`](crate::encoder::Encoder::encode) is used.
    ///
    /// Frames come back in presentation order. The count per call varies, since
    /// a picture may be held back until later ones are decoded, so drain the
    /// remainder with [`flush`](Self::flush) at end of stream. A stream without
    /// B-frames holds nothing back and yields one frame per coded frame fed.
    ///
    /// This call blocks in one case: when every output slot is held by a frame
    /// the caller has not dropped. See
    /// [`DecodeConfig::with_output_depth`].
    pub fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeFuture> {
        self.0.decode(data, pts)
    }

    /// Return any frames still held for display-order reordering.
    ///
    /// Call once after the last [`decode`](Self::decode) to emit the tail of
    /// the stream. Frames come back in presentation order, through a future that
    /// resolves behind every decode already in flight. Harmless (resolves empty)
    /// for a stream that held nothing back.
    pub fn flush(&mut self) -> Result<DecodeFuture> {
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
