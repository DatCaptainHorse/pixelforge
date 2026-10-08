//! H.265 decoding on top of Vulkan Video.
//!
//! The parser does the host-side syntax work; this module owns the Vulkan
//! session, the decoded picture buffer and the per-picture record/submit flow.

mod dpb;
mod params;
pub(crate) mod parser;
mod session;

/// Parser tests against real x265 streams (host-side, no GPU).
#[cfg(test)]
mod parser_tests;

use ash::vk;

use crate::decoder::{DecodeStatus, FrameReceiver};
use crate::error::Result;

pub(crate) use session::H265Decoder;

/// Byte offsets at which each coded picture begins in an Annex B stream.
///
/// A picture begins at each VCL NAL unit whose `first_slice_segment_in_pic_flag`
/// is set. Parameter sets, SEI and access unit delimiters are attached to the
/// picture that follows them, so the offsets carve the stream up without loss.
pub(crate) fn picture_starts(stream: &[u8]) -> Vec<usize> {
    let mut starts: Vec<usize> = Vec::new();
    let mut pending_start: Option<usize> = None;

    for (offset, nal) in parser::iter_nal_units_with_offsets(stream) {
        if nal.is_slice() {
            // first_slice_segment_in_pic_flag is the first bit of the header.
            let first_slice = nal.payload().first().is_some_and(|b| b & 0x80 != 0);
            if first_slice {
                starts.push(pending_start.take().unwrap_or(offset));
            }
            pending_start = None;
        } else if pending_start.is_none() {
            pending_start = Some(offset);
        }
    }
    starts
}

/// How many leading bytes of `buffer` hold pictures known to be complete.
pub(crate) fn complete_prefix(buffer: &[u8]) -> usize {
    picture_starts(buffer).last().copied().unwrap_or(0)
}

impl crate::decoder::DecoderApi for H265Decoder {
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        H265Decoder::decode(self, data, pts)
    }

    fn finish(&mut self) -> Result<()> {
        H265Decoder::finish(self)
    }

    fn take_frame_receiver(&mut self) -> Option<FrameReceiver> {
        H265Decoder::take_frame_receiver(self)
    }

    fn picture_format(&self) -> Option<vk::Format> {
        H265Decoder::picture_format(self)
    }
}
