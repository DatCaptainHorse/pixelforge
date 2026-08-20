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
    /// Presentation order (default). Frames come out sorted by
    /// [`display_order`](DecodedFrame::display_order), matching how a player
    /// would show them. The decoder buffers a bounded number of frames (the
    /// stream's own reorder depth), so call [`Decoder::flush`] at end of stream
    /// to drain the rest. For streams without B-frames this adds no latency and
    /// behaves like [`Self::Decode`].
    Display,
    /// Decode order: frames are returned the moment their GPU work completes,
    /// the lowest-latency option. With B-frames the caller must reorder by
    /// [`DecodedFrame::display_order`] if presentation order matters.
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
/// [`image`](Self::image) is a decoder-owned GPU image whose storage the frame
/// keeps reserved: in [`OutputOrder::Display`] it stays valid and unmodified for
/// as long as the frame is alive, and dropping the frame returns the image to
/// the decoder for reuse. Hold frames only as long as needed, and drop every
/// frame before the [`Decoder`] itself.
///
/// In [`OutputOrder::Decode`] the frame points straight at the decoder's DPB
/// image with no copy, and is only valid until the next [`Decoder::decode`] or
/// [`Decoder::flush`] call. For retention there, use [`Decoder::download`] or
/// copy it into an image you own.
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
    /// it restarts at every keyframe. In [`OutputOrder::Decode`] mode, sort by
    /// this for presentation order.
    pub display_order: i32,
    /// Whether this frame is a keyframe: a random-access point starting a new
    /// coded video sequence (H.264 IDR, H.265 IRAP, AV1 key frame).
    pub is_keyframe: bool,
    /// Keeps this frame's storage reserved; releases it on drop. `None` for a
    /// frame that borrows the decoder's DPB image (see the validity rules).
    // Held purely for its `Drop`, which is what returns the storage.
    #[allow(dead_code)]
    pub(crate) pin: Option<codec::FramePin>,
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
    ///     for frame in decoder.decode(unit, i as u64)? {
    ///         // ... use frame before the next decode() call ...
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
    /// Returns decoded frames in the configured [`OutputOrder`] (display order
    /// by default). In display order the count returned per call varies, since a
    /// picture may be held back until later ones are decoded, so drain the
    /// remainder with [`flush`](Self::flush) at end of stream. In decode order
    /// it is usually zero or one frame per coded frame fed.
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
