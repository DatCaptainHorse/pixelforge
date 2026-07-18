//! H.264 decoding on top of Vulkan Video.
//!
//! [`parser`] does the host-side syntax work; this module owns the Vulkan
//! session, the decoded picture buffer and the per-picture record/submit flow.

pub(crate) mod parser;

/// Parser tests against real x264 streams (host-side, no GPU). In-crate so they
/// reach parser internals directly, compiled only under `cargo test`.
#[cfg(test)]
mod parser_tests;

mod dpb;
mod session;

use crate::decoder::{DecodedFrame, DecodedFrameData};
use crate::error::Result;
use ash::vk;

pub(crate) use session::H264Decoder;

impl crate::decoder::DecoderApi for H264Decoder {
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<Vec<DecodedFrame>> {
        H264Decoder::decode(self, data, pts)
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        H264Decoder::flush(self)
    }

    fn download(&mut self, frame: &DecodedFrame) -> Result<DecodedFrameData> {
        H264Decoder::download(self, frame)
    }

    fn copy_frame_to_planes(
        &mut self,
        frame: &DecodedFrame,
        y_image: vk::Image,
        uv_image: vk::Image,
    ) -> Result<()> {
        H264Decoder::copy_frame_to_planes(self, frame, y_image, uv_image)
    }

    fn picture_format(&self) -> Option<vk::Format> {
        H264Decoder::picture_format(self)
    }
}
