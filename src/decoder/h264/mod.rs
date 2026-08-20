//! H.264 decoding on top of Vulkan Video.
//!
//! parser does the host-side syntax work; this module owns the Vulkan
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

/// Split an Annex B stream into one slice per coded picture.
///
/// A new picture begins at each VCL NAL whose `first_mb_in_slice` is 0.
/// Parameter sets, SEI and access unit delimiters are attached to the picture
/// that follows them, so the units can be fed back in order without loss.
pub(crate) fn split_stream(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts: Vec<usize> = Vec::new();
    let mut pending_start: Option<usize> = None;

    for (offset, nal) in parser::iter_nal_units_with_offsets(stream) {
        if nal.nal_type.is_slice() {
            // first_mb_in_slice is the leading ue(v) of the slice header; it is
            // zero -- encoded as a leading `1` bit -- exactly at a picture start.
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

    let ends = starts
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(stream.len()));

    starts
        .iter()
        .copied()
        .zip(ends)
        .map(|(start, end)| &stream[start..end])
        .collect()
}

impl crate::decoder::DecoderApi for H264Decoder {
    fn split_stream<'a>(&self, stream: &'a [u8]) -> Vec<&'a [u8]> {
        split_stream(stream)
    }

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
