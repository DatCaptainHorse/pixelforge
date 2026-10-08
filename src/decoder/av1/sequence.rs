//! The AV1 sequence header OBU and the OBU framing around it.

use crate::decoder::av1::reader::BitReader;
use crate::error::{PixelForgeError, Result};

fn invalid(what: &str) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("AV1 decode: {what}"))
}

/// OBU types (Table 6.2.2).
pub(crate) mod obu_type {
    pub const SEQUENCE_HEADER: u8 = 1;
    pub const TEMPORAL_DELIMITER: u8 = 2;
    pub const FRAME_HEADER: u8 = 3;
    pub const TILE_GROUP: u8 = 4;
    pub const FRAME: u8 = 6;
}

/// `seq_force_screen_content_tools` and `seq_force_integer_mv` when the
/// choice is left to each frame.
pub(crate) const SELECT: u8 = 2;

/// One OBU: its header and the payload that follows it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Obu<'a> {
    pub obu_type: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub has_extension: bool,
    pub payload: &'a [u8],
}

/// Split a low-overhead format stream (Section 5) into OBUs.
///
/// Every OBU has to carry its size, which is what separates this format from
/// Annex B and what lets a stream be cut anywhere between OBUs. An OBU without
/// a size field cannot end before the data does, so it is taken to be the last.
pub(crate) fn split_obus(data: &[u8]) -> Result<Vec<Obu<'_>>> {
    let mut obus = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let header = data[pos];
        if header & 0x80 != 0 {
            return Err(invalid("OBU forbidden bit is set"));
        }
        let obu_type = (header >> 3) & 0xF;
        let has_extension = header & 0x04 != 0;
        let has_size = header & 0x02 != 0;
        pos += 1;
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if has_extension {
            let ext = *data
                .get(pos)
                .ok_or_else(|| invalid("OBU extension byte missing"))?;
            temporal_id = ext >> 5;
            spatial_id = (ext >> 3) & 3;
            pos += 1;
        }
        let size = if has_size {
            let mut r = BitReader::new(&data[pos..]);
            let size = r.leb128()? as usize;
            pos += r.position() / 8;
            size
        } else {
            data.len() - pos
        };
        let end = pos
            .checked_add(size)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| invalid("OBU extends past the data"))?;
        obus.push(Obu {
            obu_type,
            temporal_id,
            spatial_id,
            has_extension,
            payload: &data[pos..end],
        });
        pos = end;
    }
    Ok(obus)
}

/// Everything about a sequence header that decoding needs.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info: Option<TimingInfo>,
    pub decoder_model_info: Option<DecoderModelInfo>,
    pub initial_display_delay_present_flag: bool,
    pub operating_points: Vec<OperatingPoint>,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits`, zero when order hints are off.
    pub order_hint_bits: u32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color: ColorConfig,
    pub film_grain_params_present: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture_minus_1: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct DecoderModelInfo {
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct OperatingPoint {
    pub idc: u16,
    pub decoder_model_present_for_this_op: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct ColorConfig {
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

impl SequenceHeader {
    pub fn num_planes(&self) -> u32 {
        if self.color.mono_chrome { 1 } else { 3 }
    }

    /// Bits of a frame id, `idLen`.
    pub fn frame_id_length(&self) -> u32 {
        self.additional_frame_id_length_minus_1 as u32
            + self.delta_frame_id_length_minus_2 as u32
            + 3
    }

    /// The operating point that is decoded: the first, which by convention is
    /// the one with the most layers.
    pub fn operating_point_idc(&self) -> u16 {
        self.operating_points.first().map_or(0, |op| op.idc)
    }
}

pub(crate) fn parse_sequence_header(payload: &[u8]) -> Result<SequenceHeader> {
    let mut r = BitReader::new(payload);
    let mut s = SequenceHeader {
        seq_profile: r.bits(3)? as u8,
        ..Default::default()
    };
    if s.seq_profile > 2 {
        return Err(invalid("unknown seq_profile"));
    }
    s.still_picture = r.flag()?;
    s.reduced_still_picture_header = r.flag()?;

    if s.reduced_still_picture_header {
        r.skip(5)?; // seq_level_idx[0]
        s.operating_points.push(OperatingPoint::default());
    } else {
        if r.flag()? {
            let mut timing = TimingInfo {
                num_units_in_display_tick: r.bits(32)?,
                time_scale: r.bits(32)?,
                equal_picture_interval: r.flag()?,
                num_ticks_per_picture_minus_1: 0,
            };
            if timing.equal_picture_interval {
                timing.num_ticks_per_picture_minus_1 = r.uvlc()?;
            }
            s.timing_info = Some(timing);
            if r.flag()? {
                s.decoder_model_info = Some(DecoderModelInfo {
                    buffer_delay_length_minus_1: r.bits(5)? as u8,
                    num_units_in_decoding_tick: r.bits(32)?,
                    buffer_removal_time_length_minus_1: r.bits(5)? as u8,
                    frame_presentation_time_length_minus_1: r.bits(5)? as u8,
                });
            }
        }
        s.initial_display_delay_present_flag = r.flag()?;
        let count = r.bits(5)? + 1;
        for _ in 0..count {
            let mut op = OperatingPoint {
                idc: r.bits(12)? as u16,
                decoder_model_present_for_this_op: false,
            };
            let seq_level_idx = r.bits(5)?;
            if seq_level_idx > 7 {
                r.skip(1)?; // seq_tier
            }
            if let Some(model) = &s.decoder_model_info {
                op.decoder_model_present_for_this_op = r.flag()?;
                if op.decoder_model_present_for_this_op {
                    let n = model.buffer_delay_length_minus_1 as usize + 1;
                    r.skip(2 * n + 1)?; // decoder and encoder buffer delay, low_delay_mode_flag
                }
            }
            if s.initial_display_delay_present_flag && r.flag()? {
                r.skip(4)?;
            }
            s.operating_points.push(op);
        }
    }

    s.frame_width_bits_minus_1 = r.bits(4)? as u8;
    s.frame_height_bits_minus_1 = r.bits(4)? as u8;
    s.max_frame_width_minus_1 = r.bits(s.frame_width_bits_minus_1 as u32 + 1)? as u16;
    s.max_frame_height_minus_1 = r.bits(s.frame_height_bits_minus_1 as u32 + 1)? as u16;
    if !s.reduced_still_picture_header {
        s.frame_id_numbers_present_flag = r.flag()?;
    }
    if s.frame_id_numbers_present_flag {
        s.delta_frame_id_length_minus_2 = r.bits(4)? as u8;
        s.additional_frame_id_length_minus_1 = r.bits(3)? as u8;
        if s.frame_id_length() > 16 {
            return Err(invalid("frame id longer than 16 bits"));
        }
    }
    s.use_128x128_superblock = r.flag()?;
    s.enable_filter_intra = r.flag()?;
    s.enable_intra_edge_filter = r.flag()?;
    if s.reduced_still_picture_header {
        s.seq_force_screen_content_tools = SELECT;
        s.seq_force_integer_mv = SELECT;
    } else {
        s.enable_interintra_compound = r.flag()?;
        s.enable_masked_compound = r.flag()?;
        s.enable_warped_motion = r.flag()?;
        s.enable_dual_filter = r.flag()?;
        s.enable_order_hint = r.flag()?;
        if s.enable_order_hint {
            s.enable_jnt_comp = r.flag()?;
            s.enable_ref_frame_mvs = r.flag()?;
        }
        s.seq_force_screen_content_tools = if r.flag()? { SELECT } else { r.bit()? as u8 };
        s.seq_force_integer_mv = if s.seq_force_screen_content_tools > 0 {
            if r.flag()? { SELECT } else { r.bit()? as u8 }
        } else {
            SELECT
        };
        if s.enable_order_hint {
            s.order_hint_bits = r.bits(3)? + 1;
        }
    }
    s.enable_superres = r.flag()?;
    s.enable_cdef = r.flag()?;
    s.enable_restoration = r.flag()?;
    s.color = parse_color_config(&mut r, s.seq_profile)?;
    s.film_grain_params_present = r.flag()?;
    Ok(s)
}

fn parse_color_config(r: &mut BitReader, seq_profile: u8) -> Result<ColorConfig> {
    let mut c = ColorConfig::default();
    let high_bitdepth = r.flag()?;
    c.bit_depth = if seq_profile == 2 && high_bitdepth {
        if r.flag()? { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    c.mono_chrome = if seq_profile == 1 { false } else { r.flag()? };
    c.color_description_present_flag = r.flag()?;
    if c.color_description_present_flag {
        c.color_primaries = r.bits(8)? as u8;
        c.transfer_characteristics = r.bits(8)? as u8;
        c.matrix_coefficients = r.bits(8)? as u8;
    } else {
        // CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED.
        c.color_primaries = 2;
        c.transfer_characteristics = 2;
        c.matrix_coefficients = 2;
    }
    if c.mono_chrome {
        c.color_range = r.flag()?;
        c.subsampling_x = true;
        c.subsampling_y = true;
        return Ok(c);
    }
    if c.color_primaries == 1 && c.transfer_characteristics == 13 && c.matrix_coefficients == 0 {
        // sRGB: full range, no subsampling.
        c.color_range = true;
    } else {
        c.color_range = r.flag()?;
        match seq_profile {
            0 => {
                c.subsampling_x = true;
                c.subsampling_y = true;
            }
            1 => {}
            _ if c.bit_depth == 12 => {
                c.subsampling_x = r.flag()?;
                c.subsampling_y = if c.subsampling_x { r.flag()? } else { false };
            }
            _ => c.subsampling_x = true,
        }
        if c.subsampling_x && c.subsampling_y {
            c.chroma_sample_position = r.bits(2)? as u8;
        }
    }
    c.separate_uv_delta_q = r.flag()?;
    Ok(c)
}
