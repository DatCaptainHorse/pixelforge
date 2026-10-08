//! AV1 decoding on top of Vulkan Video.
//!
//! The parser does the host-side syntax work: unlike H.26x, an AV1 decoder
//! hands the driver the whole frame header as parsed fields, so the header is
//! parsed in full. This module owns the Vulkan session, the picture buffer and
//! the per-frame record/submit flow.

mod dpb;
mod frame_header;
mod params;
mod reader;
mod sequence;
mod session;

/// Parser tests against real libaom and SVT-AV1 streams (host-side, no GPU).
#[cfg(test)]
mod parser_tests;

use ash::vk;

use crate::decoder::{DecodeStatus, FrameReceiver};
use crate::error::Result;

pub(crate) use session::Av1Decoder;

/// Byte offsets at which each temporal unit begins in a low-overhead OBU
/// stream: the temporal delimiter OBUs.
///
/// Stops at an OBU that does not fit in `stream`, which is how a stream cut
/// mid-OBU ends.
pub(crate) fn temporal_unit_starts(stream: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pos = 0usize;
    while pos < stream.len() {
        let header = stream[pos];
        let has_extension = header & 0x04 != 0;
        let has_size = header & 0x02 != 0;
        if !has_size {
            break;
        }
        let mut cursor = pos + 1 + has_extension as usize;
        let mut size = 0usize;
        let mut complete = false;
        for i in 0..8 {
            let Some(&byte) = stream.get(cursor) else {
                break;
            };
            cursor += 1;
            size |= ((byte & 0x7F) as usize) << (7 * i);
            if byte & 0x80 == 0 {
                complete = true;
                break;
            }
        }
        if !complete || cursor + size > stream.len() {
            break;
        }
        if (header >> 3) & 0xF == sequence::obu_type::TEMPORAL_DELIMITER {
            starts.push(pos);
        }
        pos = cursor + size;
    }
    starts
}

/// How many leading bytes of `buffer` hold temporal units known to be
/// complete: everything before the last temporal delimiter.
pub(crate) fn complete_prefix(buffer: &[u8]) -> usize {
    temporal_unit_starts(buffer).last().copied().unwrap_or(0)
}

impl crate::decoder::DecoderApi for Av1Decoder {
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        Av1Decoder::decode(self, data, pts)
    }

    fn finish(&mut self) -> Result<()> {
        Av1Decoder::finish(self)
    }

    fn take_frame_receiver(&mut self) -> Option<FrameReceiver> {
        Av1Decoder::take_frame_receiver(self)
    }

    fn picture_format(&self) -> Option<vk::Format> {
        Av1Decoder::picture_format(self)
    }
}
