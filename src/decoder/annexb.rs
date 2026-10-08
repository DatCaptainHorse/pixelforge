//! Splitting an Annex B byte stream into NAL units, shared by the codecs that
//! use one (H.264 and H.265).

/// Find the next start code at or after `from`.
///
/// Returns the position of the start code and the position of the byte after
/// it. Accepts the 3-byte and 4-byte forms.
pub(crate) fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, i + 3));
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, i + 4));
            }
        }
        i += 1;
    }
    None
}

/// The NAL units of an Annex B stream, each with the offset of its start code.
///
/// A NAL unit is returned without its start code and without the zero padding
/// that may precede the next one.
pub(crate) fn iter_nals(data: &[u8]) -> impl Iterator<Item = (usize, &[u8])> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        loop {
            let (start_code_pos, payload_start) = find_start_code(data, pos)?;
            let end = match find_start_code(data, payload_start) {
                Some((next_start, _)) => next_start,
                None => data.len(),
            };
            pos = end;
            let mut trimmed_end = end;
            while trimmed_end > payload_start && data[trimmed_end - 1] == 0 {
                trimmed_end -= 1;
            }
            if trimmed_end > payload_start {
                return Some((start_code_pos, &data[payload_start..trimmed_end]));
            }
        }
    })
}
