//! Hardware-accelerated video decoding using Vulkan Video.
//!
//! The decoder consumes a coded byte stream and produces decoded frames as GPU
//! images (`vk::Image`), optionally with CPU readback. [`Decoder::decode`]
//! takes whatever the caller has: whole coded frames, if they arrive already
//! framed by a container or transport, or an arbitrary slice of a byte stream,
//! which the decoder frames itself using the codec's own rules (Annex B start
//! codes for H.264/H.265, OBU temporal units for AV1). See [`Framing`].
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

/// How the bytes handed to [`Decoder::decode`] are framed.
///
/// The decoder has to know where one coded frame ends and the next begins.
/// Which of these applies is a property of where the bytes come from, not of
/// the codec, so it cannot be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// Every call carries whole coded frames: the end of the data is the end of
    /// a frame (the default).
    ///
    /// This is what a container or a transport gives you, where framing has
    /// already been done: an RTP access unit, an MP4 sample, a WebRTC frame.
    /// Nothing is buffered, so a frame is decoded by the call that delivers it
    /// and this is the lower-latency option.
    FrameAligned,
    /// A continuous byte stream that may cut anywhere, including mid-frame.
    ///
    /// This is what a raw `.264` file or a socket carrying one gives you. The
    /// decoder buffers a trailing partial frame until later bytes complete it,
    /// which costs one frame of latency: a coded frame cannot be known to be
    /// complete until the start of the next one is seen. [`Decoder::flush`]
    /// decodes whatever is still buffered, so the last frame is not lost.
    ByteStream,
}

/// How many decoded frames may be outstanding at once, by default.
///
/// Two for the caller to hold while consuming them, matching the encoder's
/// pipeline depth, plus two for the frames the decode pipeline itself has in
/// flight between submitting a picture and delivering it.
///
/// A caller feeding several coded frames per [`Decoder::decode`] call needs
/// more, since every frame that call emits is outstanding at once. See
/// [`DecodeConfig::with_output_depth`].
pub const DEFAULT_OUTPUT_DEPTH: usize = 4;

/// Configuration for creating a [`Decoder`].
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// The codec to decode.
    pub codec: Codec,
    /// How many decoded frames the caller may hold at once. Defaults to
    /// [`DEFAULT_OUTPUT_DEPTH`]. See [`DecodeConfig::with_output_depth`].
    pub output_depth: usize,
    /// Queue family that will read decoded frames, if it is not the decoder's
    /// own. See [`DecodeConfig::with_consumer_queue_family`].
    pub consumer_queue_family: Option<u32>,
    /// How input handed to [`Decoder::decode`] is framed. Defaults to
    /// [`Framing::FrameAligned`].
    pub framing: Framing,
}

impl DecodeConfig {
    /// Create an H.264 decode configuration.
    pub fn h264() -> Self {
        Self {
            codec: Codec::H264,
            output_depth: DEFAULT_OUTPUT_DEPTH,
            consumer_queue_family: None,
            framing: Framing::FrameAligned,
        }
    }

    /// How many decoded frames the caller may hold at once.
    ///
    /// A frame is the decoder's own DPB image, so an outstanding frame keeps a
    /// DPB slot reserved. The decoder allocates this many slots beyond what the
    /// stream's references and reorder depth need.
    ///
    /// Exceeding it costs a copy, not an error: pictures past the budget are
    /// copied into private images so decoding can continue. So this is a
    /// performance setting, and the number to match is how many frames are
    /// outstanding at once, which is everything a [`Decoder::decode`] call
    /// emits plus whatever the caller is still holding. Feeding one coded frame
    /// per call keeps that small; feeding 64 KB of byte stream at a time does
    /// not.
    ///
    /// Clamped to what the device's DPB slot limit allows for the stream, so a
    /// device too small for the request degrades to copying rather than
    /// failing. Each slot costs one decoded picture's worth of memory.
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

    /// Name the queue family that will read decoded frames.
    ///
    /// A [`DecodedFrame`] is a decoder-owned image, and images are shared
    /// between queue families explicitly. By default the decoder shares its
    /// pictures with its own decode and transfer families only, which is all
    /// [`Decoder::download`] needs. A renderer sampling frames from a graphics
    /// queue is a *third* family, and reading an image from a family it was not
    /// shared with is undefined behaviour unless the caller performs a queue
    /// family ownership transfer themselves.
    ///
    /// Passing that family here adds it to the sharing set, so frames can be
    /// sampled directly with no ownership transfer and no copy. Set it whenever
    /// frames are consumed by anything other than
    /// [`Decoder::download`]; for a context adopted from an existing device
    /// (see
    /// [`build_from_existing_decode`](crate::vulkan::VideoContextBuilder::build_from_existing_decode))
    /// this is the family the caller already renders on.
    pub fn with_consumer_queue_family(mut self, family: u32) -> Self {
        self.consumer_queue_family = Some(family);
        self
    }

    /// Feed a continuous byte stream that may cut mid-frame, rather than whole
    /// coded frames per call. See [`Framing::ByteStream`].
    pub fn with_byte_stream(mut self) -> Self {
        self.framing = Framing::ByteStream;
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
///   [`DecodeConfig::with_output_depth`] spare slots. Once they are all
///   reserved the decoder falls back to copying pictures out, which still
///   works but gives up the zero-copy path. Drop frames promptly.
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
    ///
    /// Created with `TRANSFER_SRC` and, where the device allows it,
    /// `SAMPLED` (see [`sampleable`](Self::sampleable)). Reading it from a
    /// queue family other than the decoder's own requires
    /// [`DecodeConfig::with_consumer_queue_family`].
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
    /// Whether [`image`](Self::image) was created with `SAMPLED` usage, so a
    /// shader can read it directly rather than copying from it first.
    ///
    /// True on every driver tested so far. It is false only where the device
    /// reports no usable picture format for a sampleable decode image, in which
    /// case the frame can still be copied from (`TRANSFER_SRC`).
    ///
    /// Sampling also needs a view of the caller's own: the format is
    /// multi-planar YCbCr, so it takes a `VkSamplerYcbcrConversion`, and the
    /// image must be transitioned out of [`layout`](Self::layout) first.
    pub sampleable: bool,
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

    /// Decode a chunk of the coded byte stream.
    ///
    /// What `data` may contain depends on the configured [`Framing`]. By
    /// default ([`Framing::FrameAligned`]) it must hold whole coded frames:
    /// any number of them, plus whatever non-picture data the codec carries
    /// alongside (H.26x parameter sets and SEI, AV1 sequence headers), ending
    /// on a frame boundary. Feed exactly one coded frame per call for the
    /// lowest latency; it is decoded by this call.
    ///
    /// With [`Framing::ByteStream`] the data may cut anywhere and the decoder
    /// does the framing itself, holding back a trailing partial frame until
    /// later calls complete it.
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
    /// Drop frames promptly. A frame the caller holds keeps a DPB slot, and
    /// once the slots run out the decoder copies pictures out instead of
    /// handing over its own images. See [`DecodeConfig::with_output_depth`].
    pub fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeFuture> {
        self.0.decode(data, pts)
    }

    /// Return any frames still held back: the reorder buffer's contents, and in
    /// [`Framing::ByteStream`] the trailing frame no later data will complete.
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
