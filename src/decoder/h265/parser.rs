//! Minimal H.265 Annex B parsing: NAL unit splitting plus VPS, SPS, PPS and the
//! start of the slice segment header.
//!
//! Like the H.264 parser this is not a full parser. Vulkan Video decode takes
//! the slice segment NAL units as they are and parses the rest of the header in
//! the driver; the host needs the parameter sets to describe the stream to the
//! driver, and the leading slice header syntax to work out the picture order
//! count and which pictures the reference picture set keeps.

use crate::decoder::bitreader::BitReader;
use crate::error::{PixelForgeError, Result};

/// NAL unit type values from Table 7-1.
pub(crate) mod nal {
    pub const RADL_N: u8 = 6;
    pub const RADL_R: u8 = 7;
    pub const RASL_N: u8 = 8;
    pub const RASL_R: u8 = 9;
    pub const BLA_W_LP: u8 = 16;
    pub const BLA_N_LP: u8 = 18;
    pub const IDR_W_RADL: u8 = 19;
    pub const IDR_N_LP: u8 = 20;
    #[cfg(test)]
    pub const CRA: u8 = 21;
    pub const RSV_IRAP_VCL23: u8 = 23;
    pub const VPS: u8 = 32;
    pub const SPS: u8 = 33;
    pub const PPS: u8 = 34;
    pub const EOS: u8 = 36;
    pub const EOB: u8 = 37;
}

/// A single NAL unit within an Annex B stream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NalUnit<'a> {
    pub nal_type: u8,
    /// `TemporalId`, which is `nuh_temporal_id_plus1 - 1`.
    pub temporal_id: u8,
    /// The complete NAL unit: two header bytes, then the payload.
    pub data: &'a [u8],
}

impl<'a> NalUnit<'a> {
    /// The payload after the two-byte NAL header.
    pub fn payload(&self) -> &'a [u8] {
        &self.data[2..]
    }

    /// A video coding layer NAL unit: a slice segment of some kind.
    pub fn is_slice(&self) -> bool {
        self.nal_type < 32
    }

    /// A slice segment of a type this decoder can decode. The reserved VCL
    /// types are not.
    pub fn is_decodable_slice(&self) -> bool {
        matches!(self.nal_type, 0..=9 | 16..=21)
    }

    pub fn is_irap(&self) -> bool {
        is_irap(self.nal_type)
    }
}

pub(crate) fn is_irap(nal_type: u8) -> bool {
    (nal::BLA_W_LP..=nal::RSV_IRAP_VCL23).contains(&nal_type)
}

pub(crate) fn is_idr(nal_type: u8) -> bool {
    matches!(nal_type, nal::IDR_W_RADL | nal::IDR_N_LP)
}

pub(crate) fn is_bla(nal_type: u8) -> bool {
    (nal::BLA_W_LP..=nal::BLA_N_LP).contains(&nal_type)
}

pub(crate) fn is_rasl(nal_type: u8) -> bool {
    matches!(nal_type, nal::RASL_N | nal::RASL_R)
}

pub(crate) fn is_radl(nal_type: u8) -> bool {
    matches!(nal_type, nal::RADL_N | nal::RADL_R)
}

/// A sub-layer non-reference picture: the even types below 16.
pub(crate) fn is_sub_layer_non_reference(nal_type: u8) -> bool {
    nal_type < 16 && nal_type.is_multiple_of(2)
}

/// Iterate over the NAL units of an Annex B stream, with the offset of each
/// one's start code. NAL units of layers above the base layer are skipped.
pub(crate) fn iter_nal_units_with_offsets(
    data: &[u8],
) -> impl Iterator<Item = (usize, NalUnit<'_>)> {
    crate::decoder::annexb::iter_nals(data).filter_map(|(offset, nal)| {
        // An end of sequence NAL unit is nothing but its two header bytes.
        if nal.len() < 2 {
            return None;
        }
        let layer_id = ((nal[0] & 1) << 5) | (nal[1] >> 3);
        if layer_id != 0 {
            return None;
        }
        let temporal_id_plus1 = nal[1] & 7;
        if temporal_id_plus1 == 0 {
            return None;
        }
        Some((
            offset,
            NalUnit {
                nal_type: (nal[0] >> 1) & 0x3F,
                temporal_id: temporal_id_plus1 - 1,
                data: nal,
            },
        ))
    })
}

pub(crate) fn iter_nal_units(data: &[u8]) -> impl Iterator<Item = NalUnit<'_>> {
    iter_nal_units_with_offsets(data).map(|(_, nal)| nal)
}

fn invalid(what: &str) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("H.265 decode: {what}"))
}

/// `Ceil(Log2(x))`.
fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// `profile_tier_level()`, general part only; sub-layer parts are read and
/// dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ProfileTierLevel {
    pub tier_flag: bool,
    pub profile_idc: u8,
    pub progressive_source_flag: bool,
    pub interlaced_source_flag: bool,
    pub non_packed_constraint_flag: bool,
    pub frame_only_constraint_flag: bool,
    pub level_idc: u8,
}

fn parse_profile_tier_level(
    r: &mut BitReader,
    max_sub_layers_minus1: u32,
) -> Result<ProfileTierLevel> {
    let _profile_space = r.bits(2)?;
    let tier_flag = r.flag()?;
    let profile_idc = r.bits(5)? as u8;
    let _compatibility_flags = r.bits(32)?;
    let progressive_source_flag = r.flag()?;
    let interlaced_source_flag = r.flag()?;
    let non_packed_constraint_flag = r.flag()?;
    let frame_only_constraint_flag = r.flag()?;
    // 43 bits of constraint flags or reserved bits, then one more.
    r.skip(44)?;
    let level_idc = r.bits(8)? as u8;

    let mut profile_present = [false; 8];
    let mut level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        profile_present[i] = r.flag()?;
        level_present[i] = r.flag()?;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.skip(2)?;
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if profile_present[i] {
            r.skip(88)?;
        }
        if level_present[i] {
            r.skip(8)?;
        }
    }
    Ok(ProfileTierLevel {
        tier_flag,
        profile_idc,
        progressive_source_flag,
        interlaced_source_flag,
        non_packed_constraint_flag,
        frame_only_constraint_flag,
        level_idc,
    })
}

/// The per-sub-layer DPB sizing a VPS or SPS declares.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct DecPicBuf {
    pub max_dec_pic_buffering_minus1: [u8; 7],
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u32; 7],
}

/// The sub-layer ordering info of a VPS or SPS. Returns it along with whether
/// it was sent for every sub-layer; when it was not, the one value sent is the
/// highest sub-layer's and applies to all of them.
fn parse_dec_pic_buf(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<(DecPicBuf, bool)> {
    let present = r.flag()?;
    let mut buf = DecPicBuf::default();
    let first = if present { 0 } else { max_sub_layers_minus1 };
    for i in first..=max_sub_layers_minus1 {
        let i = i as usize;
        buf.max_dec_pic_buffering_minus1[i] = r.ue()?.min(255) as u8;
        buf.max_num_reorder_pics[i] = r.ue()?.min(255) as u8;
        buf.max_latency_increase_plus1[i] = r.ue()?;
    }
    if !present {
        let top = max_sub_layers_minus1 as usize;
        for i in 0..top {
            buf.max_dec_pic_buffering_minus1[i] = buf.max_dec_pic_buffering_minus1[top];
            buf.max_num_reorder_pics[i] = buf.max_num_reorder_pics[top];
            buf.max_latency_increase_plus1[i] = buf.max_latency_increase_plus1[top];
        }
    }
    Ok((buf, present))
}

/// Parsed video parameter set (the subset Vulkan decode needs).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Vps {
    pub vps_id: u8,
    pub max_sub_layers_minus1: u8,
    pub temporal_id_nesting_flag: bool,
    pub ptl: ProfileTierLevel,
    pub dec_pic_buf: DecPicBuf,
    pub sub_layer_ordering_info_present_flag: bool,
}

pub(crate) fn parse_vps(payload: &[u8]) -> Result<Vps> {
    let mut r = BitReader::new(payload);
    let vps_id = r.bits(4)? as u8;
    r.skip(2)?; // base layer internal / available
    r.skip(6)?; // vps_max_layers_minus1
    let max_sub_layers_minus1 = r.bits(3)?;
    let temporal_id_nesting_flag = r.flag()?;
    r.skip(16)?; // vps_reserved_0xffff_16bits
    let ptl = parse_profile_tier_level(&mut r, max_sub_layers_minus1)?;
    let (dec_pic_buf, sub_layer_ordering_info_present_flag) =
        parse_dec_pic_buf(&mut r, max_sub_layers_minus1)?;
    Ok(Vps {
        vps_id,
        max_sub_layers_minus1: max_sub_layers_minus1 as u8,
        temporal_id_nesting_flag,
        ptl,
        dec_pic_buf,
        sub_layer_ordering_info_present_flag,
    })
}

/// Scaling lists in the order the syntax codes them, which is also the order
/// Vulkan takes them in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalingLists {
    pub list_4x4: [[u8; 16]; 6],
    pub list_8x8: [[u8; 64]; 6],
    pub list_16x16: [[u8; 64]; 6],
    pub list_32x32: [[u8; 64]; 2],
    pub dc_16x16: [u8; 6],
    pub dc_32x32: [u8; 2],
}

/// The default 8x8 intra list (Table 7-6), as an 8x8 matrix in raster order.
#[rustfmt::skip]
const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24,
    16, 16, 16, 16, 17, 19, 22, 25,
    16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36,
    17, 17, 20, 24, 30, 35, 41, 47,
    18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88,
    24, 25, 29, 36, 47, 65, 88, 115,
];

/// The default 8x8 inter list, as an 8x8 matrix in raster order.
#[rustfmt::skip]
const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24,
    16, 16, 16, 17, 18, 20, 24, 25,
    16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33,
    17, 18, 20, 24, 25, 28, 33, 41,
    18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71,
    24, 25, 28, 33, 41, 54, 71, 91,
];

/// An 8x8 matrix in raster order, as a list in up-right diagonal scan order,
/// which is how the syntax codes scaling lists.
fn raster_to_scan(matrix: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let (mut i, mut x, mut y) = (0usize, 0i32, 0i32);
    while i < 64 {
        while y >= 0 {
            if x < 8 && y < 8 {
                out[i] = matrix[(y * 8 + x) as usize];
                i += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    out
}

impl ScalingLists {
    /// The lists in force when a stream asks for scaling lists without sending
    /// any: flat for 4x4, the defaults for everything larger.
    pub fn defaults() -> Self {
        let intra = raster_to_scan(&DEFAULT_INTRA);
        let inter = raster_to_scan(&DEFAULT_INTER);
        let pick = |matrix_id: usize| if matrix_id < 3 { intra } else { inter };
        Self {
            list_4x4: [[16; 16]; 6],
            list_8x8: std::array::from_fn(pick),
            list_16x16: std::array::from_fn(pick),
            list_32x32: [intra, inter],
            dc_16x16: [16; 6],
            dc_32x32: [16; 2],
        }
    }
}

/// `scaling_list_data()`.
fn parse_scaling_list_data(r: &mut BitReader) -> Result<ScalingLists> {
    let mut lists = ScalingLists::defaults();
    for size_id in 0..4usize {
        let step = if size_id == 3 { 3 } else { 1 };
        let mut matrix_id = 0usize;
        while matrix_id < 6 {
            let pred_mode_flag = r.flag()?;
            if !pred_mode_flag {
                let delta = r.ue()? as usize * step;
                if delta > matrix_id {
                    return Err(invalid("scaling_list_pred_matrix_id_delta out of range"));
                }
                if delta == 0 {
                    // The default list, which `defaults` already put there but
                    // an earlier explicit list may have been parsed over.
                    let d = ScalingLists::defaults();
                    match size_id {
                        0 => lists.list_4x4[matrix_id] = d.list_4x4[matrix_id],
                        1 => lists.list_8x8[matrix_id] = d.list_8x8[matrix_id],
                        2 => {
                            lists.list_16x16[matrix_id] = d.list_16x16[matrix_id];
                            lists.dc_16x16[matrix_id] = 16;
                        }
                        _ => {
                            lists.list_32x32[matrix_id / 3] = d.list_32x32[matrix_id / 3];
                            lists.dc_32x32[matrix_id / 3] = 16;
                        }
                    }
                } else {
                    let reference = matrix_id - delta;
                    match size_id {
                        0 => lists.list_4x4[matrix_id] = lists.list_4x4[reference],
                        1 => lists.list_8x8[matrix_id] = lists.list_8x8[reference],
                        2 => {
                            lists.list_16x16[matrix_id] = lists.list_16x16[reference];
                            lists.dc_16x16[matrix_id] = lists.dc_16x16[reference];
                        }
                        _ => {
                            lists.list_32x32[matrix_id / 3] = lists.list_32x32[reference / 3];
                            lists.dc_32x32[matrix_id / 3] = lists.dc_32x32[reference / 3];
                        }
                    }
                }
            } else {
                let mut next_coef = 8i32;
                let coef_num = 64.min(1usize << (4 + (size_id << 1)));
                let mut dc = 16u8;
                if size_id > 1 {
                    let dc_minus8 = r.se()?;
                    if !(-7..=247).contains(&dc_minus8) {
                        return Err(invalid("scaling_list_dc_coef_minus8 out of range"));
                    }
                    next_coef = dc_minus8 + 8;
                    dc = next_coef as u8;
                }
                let mut list = [0u8; 64];
                for entry in list.iter_mut().take(coef_num) {
                    let delta = r.se()?;
                    next_coef = (next_coef + delta + 256).rem_euclid(256);
                    *entry = next_coef as u8;
                }
                match size_id {
                    0 => lists.list_4x4[matrix_id].copy_from_slice(&list[..16]),
                    1 => lists.list_8x8[matrix_id] = list,
                    2 => {
                        lists.list_16x16[matrix_id] = list;
                        lists.dc_16x16[matrix_id] = dc;
                    }
                    _ => {
                        lists.list_32x32[matrix_id / 3] = list;
                        lists.dc_32x32[matrix_id / 3] = dc;
                    }
                }
            }
            matrix_id += step;
        }
    }
    Ok(lists)
}

/// A short-term reference picture set, in the derived form the decoding
/// process works with, plus the syntax elements it was coded with.
///
/// Vulkan wants both: the raw syntax, so the driver can walk the slice header,
/// and the resolved deltas.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ShortTermRps {
    pub inter_ref_pic_set_prediction_flag: bool,
    pub delta_idx_minus1: u32,
    pub delta_rps_sign: bool,
    pub abs_delta_rps_minus1: u32,
    /// Bit `j` is `use_delta_flag[j]`.
    pub use_delta_flag: u16,
    /// Bit `j` is `used_by_curr_pic_flag[j]`.
    pub used_by_curr_pic_flag: u16,
    /// `NumDeltaPocs[RefRpsIdx]`: the size of the set an inter-predicted set
    /// was predicted from. Zero for a set coded directly.
    pub num_delta_pocs_of_ref_rps: u8,

    pub num_negative_pics: u8,
    pub num_positive_pics: u8,
    /// `DeltaPocS0`: negative, in decreasing POC order (closest first).
    pub delta_poc_s0: [i32; 16],
    pub used_by_curr_pic_s0: [bool; 16],
    /// `DeltaPocS1`: positive, in increasing POC order.
    pub delta_poc_s1: [i32; 16],
    pub used_by_curr_pic_s1: [bool; 16],
}

impl ShortTermRps {
    pub fn num_delta_pocs(&self) -> usize {
        self.num_negative_pics as usize + self.num_positive_pics as usize
    }
}

/// `st_ref_pic_set(stRpsIdx)`. `sets` holds the sets of the SPS already
/// parsed; `num_sets` is `num_short_term_ref_pic_sets`, which differs from
/// `stRpsIdx` only for a set coded in a slice header.
pub(super) fn parse_short_term_rps(
    r: &mut BitReader,
    idx: usize,
    num_sets: usize,
    sets: &[ShortTermRps],
) -> Result<ShortTermRps> {
    let mut rps = ShortTermRps::default();
    if idx != 0 {
        rps.inter_ref_pic_set_prediction_flag = r.flag()?;
    }
    if rps.inter_ref_pic_set_prediction_flag {
        if idx == num_sets {
            rps.delta_idx_minus1 = r.ue()?;
        }
        rps.delta_rps_sign = r.flag()?;
        rps.abs_delta_rps_minus1 = r.ue()?;
        let ref_idx = idx
            .checked_sub(rps.delta_idx_minus1 as usize + 1)
            .filter(|&i| i < sets.len())
            .ok_or_else(|| invalid("inter RPS refers to a set that does not exist"))?;
        let reference = &sets[ref_idx];
        let delta_rps = (1 - 2 * rps.delta_rps_sign as i32) * (rps.abs_delta_rps_minus1 as i32 + 1);

        let n = reference.num_delta_pocs();
        rps.num_delta_pocs_of_ref_rps = n as u8;
        let mut used = [false; 17];
        let mut use_delta = [true; 17];
        for j in 0..=n {
            used[j] = r.flag()?;
            if used[j] {
                rps.used_by_curr_pic_flag |= 1 << j;
            } else {
                use_delta[j] = r.flag()?;
            }
            if use_delta[j] {
                rps.use_delta_flag |= 1 << j;
            }
        }

        let neg = reference.num_negative_pics as usize;
        let pos = reference.num_positive_pics as usize;
        let mut i = 0usize;
        for j in (0..pos).rev() {
            let d = reference.delta_poc_s1[j] + delta_rps;
            if d < 0 && use_delta[neg + j] {
                push_delta(
                    &mut rps.delta_poc_s0,
                    &mut rps.used_by_curr_pic_s0,
                    &mut i,
                    d,
                    used[neg + j],
                )?;
            }
        }
        if delta_rps < 0 && use_delta[n] {
            push_delta(
                &mut rps.delta_poc_s0,
                &mut rps.used_by_curr_pic_s0,
                &mut i,
                delta_rps,
                used[n],
            )?;
        }
        for j in 0..neg {
            let d = reference.delta_poc_s0[j] + delta_rps;
            if d < 0 && use_delta[j] {
                push_delta(
                    &mut rps.delta_poc_s0,
                    &mut rps.used_by_curr_pic_s0,
                    &mut i,
                    d,
                    used[j],
                )?;
            }
        }
        rps.num_negative_pics = i as u8;

        let mut i = 0usize;
        for j in (0..neg).rev() {
            let d = reference.delta_poc_s0[j] + delta_rps;
            if d > 0 && use_delta[j] {
                push_delta(
                    &mut rps.delta_poc_s1,
                    &mut rps.used_by_curr_pic_s1,
                    &mut i,
                    d,
                    used[j],
                )?;
            }
        }
        if delta_rps > 0 && use_delta[n] {
            push_delta(
                &mut rps.delta_poc_s1,
                &mut rps.used_by_curr_pic_s1,
                &mut i,
                delta_rps,
                used[n],
            )?;
        }
        for j in 0..pos {
            let d = reference.delta_poc_s1[j] + delta_rps;
            if d > 0 && use_delta[neg + j] {
                push_delta(
                    &mut rps.delta_poc_s1,
                    &mut rps.used_by_curr_pic_s1,
                    &mut i,
                    d,
                    used[neg + j],
                )?;
            }
        }
        rps.num_positive_pics = i as u8;
    } else {
        let num_negative = r.ue()? as usize;
        let num_positive = r.ue()? as usize;
        if num_negative > 16 || num_positive > 16 || num_negative + num_positive > 16 {
            return Err(invalid("short-term RPS holds more than 16 pictures"));
        }
        let mut poc = 0i32;
        for i in 0..num_negative {
            poc -= r.ue()? as i32 + 1;
            rps.delta_poc_s0[i] = poc;
            rps.used_by_curr_pic_s0[i] = r.flag()?;
        }
        let mut poc = 0i32;
        for i in 0..num_positive {
            poc += r.ue()? as i32 + 1;
            rps.delta_poc_s1[i] = poc;
            rps.used_by_curr_pic_s1[i] = r.flag()?;
        }
        rps.num_negative_pics = num_negative as u8;
        rps.num_positive_pics = num_positive as u8;
    }
    Ok(rps)
}

fn push_delta(
    deltas: &mut [i32; 16],
    used: &mut [bool; 16],
    i: &mut usize,
    delta: i32,
    is_used: bool,
) -> Result<()> {
    if *i >= 16 {
        return Err(invalid("short-term RPS holds more than 16 pictures"));
    }
    deltas[*i] = delta;
    used[*i] = is_used;
    *i += 1;
    Ok(())
}

/// `hrd_parameters()`, read to be skipped: nothing in it matters to decoding,
/// but it sits in the middle of the VUI, ahead of the SPS extensions.
fn skip_hrd_parameters(
    r: &mut BitReader,
    common_inf: bool,
    max_sub_layers_minus1: u32,
) -> Result<()> {
    let mut nal_hrd = false;
    let mut vcl_hrd = false;
    let mut sub_pic = false;
    if common_inf {
        nal_hrd = r.flag()?;
        vcl_hrd = r.flag()?;
        if nal_hrd || vcl_hrd {
            sub_pic = r.flag()?;
            if sub_pic {
                r.skip(8 + 5 + 1 + 5)?;
            }
            r.skip(4 + 4)?;
            if sub_pic {
                r.skip(4)?;
            }
            r.skip(5 + 5 + 5)?;
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_general = r.flag()?;
        let fixed_within_cvs = if fixed_general { true } else { r.flag()? };
        let mut low_delay = false;
        if fixed_within_cvs {
            r.ue()?;
        } else {
            low_delay = r.flag()?;
        }
        let cpb_cnt = if low_delay { 1 } else { r.ue()? + 1 };
        for present in [nal_hrd, vcl_hrd] {
            if present {
                for _ in 0..cpb_cnt {
                    r.ue()?;
                    r.ue()?;
                    if sub_pic {
                        r.ue()?;
                        r.ue()?;
                    }
                    r.skip(1)?;
                }
            }
        }
    }
    Ok(())
}

/// `vui_parameters()`, read to be skipped.
fn skip_vui(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<()> {
    if r.flag()? {
        let aspect_ratio_idc = r.bits(8)?;
        if aspect_ratio_idc == 255 {
            r.skip(32)?;
        }
    }
    if r.flag()? {
        r.skip(1)?; // overscan_appropriate_flag
    }
    if r.flag()? {
        r.skip(4)?; // video_format, video_full_range_flag
        if r.flag()? {
            r.skip(24)?;
        }
    }
    if r.flag()? {
        r.ue()?;
        r.ue()?;
    }
    r.skip(3)?; // neutral_chroma_indication, field_seq, frame_field_info_present
    if r.flag()? {
        for _ in 0..4 {
            r.ue()?;
        }
    }
    if r.flag()? {
        r.skip(32)?;
        r.skip(32)?;
        if r.flag()? {
            r.ue()?;
        }
        if r.flag()? {
            skip_hrd_parameters(r, true, max_sub_layers_minus1)?;
        }
    }
    if r.flag()? {
        r.skip(3)?;
        for _ in 0..5 {
            r.ue()?;
        }
    }
    Ok(())
}

/// Parsed sequence parameter set (the subset Vulkan decode needs).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Sps {
    pub vps_id: u8,
    pub max_sub_layers_minus1: u8,
    pub temporal_id_nesting_flag: bool,
    pub ptl: ProfileTierLevel,
    pub sps_id: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub conformance_window_flag: bool,
    pub conf_win_left_offset: u32,
    pub conf_win_right_offset: u32,
    pub conf_win_top_offset: u32,
    pub conf_win_bottom_offset: u32,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub dec_pic_buf: DecPicBuf,
    pub sub_layer_ordering_info_present_flag: bool,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_luma_transform_block_size_minus2: u8,
    pub log2_diff_max_min_luma_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub scaling_list_enabled_flag: bool,
    pub sps_scaling_list_data_present_flag: bool,
    /// The lists in force when `scaling_list_enabled_flag` is set: the coded
    /// ones, or the defaults when none were sent.
    pub scaling_lists: Option<ScalingLists>,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub pcm_loop_filter_disabled_flag: bool,
    pub short_term_rps: Vec<ShortTermRps>,
    pub long_term_ref_pics_present_flag: bool,
    pub lt_ref_pic_poc_lsb_sps: Vec<u32>,
    pub used_by_curr_pic_lt_sps_flag: Vec<bool>,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    pub range_extension: SpsRangeExtension,
}

/// The flags of `sps_range_extension()`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SpsRangeExtension {
    pub present: bool,
    pub transform_skip_rotation_enabled_flag: bool,
    pub transform_skip_context_enabled_flag: bool,
    pub implicit_rdpcm_enabled_flag: bool,
    pub explicit_rdpcm_enabled_flag: bool,
    pub extended_precision_processing_flag: bool,
    pub intra_smoothing_disabled_flag: bool,
    pub high_precision_offsets_enabled_flag: bool,
    pub persistent_rice_adaptation_enabled_flag: bool,
    pub cabac_bypass_alignment_enabled_flag: bool,
}

impl Sps {
    pub fn log2_max_pic_order_cnt_lsb(&self) -> u32 {
        self.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4
    }

    /// `CtbLog2SizeY`.
    pub fn ctb_log2_size(&self) -> u32 {
        self.log2_min_luma_coding_block_size_minus3 as u32
            + 3
            + self.log2_diff_max_min_luma_coding_block_size as u32
    }

    pub fn pic_width_in_ctbs(&self) -> u32 {
        self.pic_width_in_luma_samples
            .div_ceil(1 << self.ctb_log2_size())
    }

    pub fn pic_height_in_ctbs(&self) -> u32 {
        self.pic_height_in_luma_samples
            .div_ceil(1 << self.ctb_log2_size())
    }

    /// `PicSizeInCtbsY`.
    pub fn pic_size_in_ctbs(&self) -> u32 {
        self.pic_width_in_ctbs() * self.pic_height_in_ctbs()
    }

    /// `SubWidthC` and `SubHeightC`.
    fn chroma_unit(&self) -> (u32, u32) {
        match (self.chroma_format_idc, self.separate_colour_plane_flag) {
            (1, false) => (2, 2),
            (2, false) => (2, 1),
            _ => (1, 1),
        }
    }

    /// The visible size, after the conformance window.
    pub fn display_dimensions(&self) -> (u32, u32) {
        let (sub_w, sub_h) = self.chroma_unit();
        let w = self
            .pic_width_in_luma_samples
            .saturating_sub(sub_w * (self.conf_win_left_offset + self.conf_win_right_offset));
        let h = self
            .pic_height_in_luma_samples
            .saturating_sub(sub_h * (self.conf_win_top_offset + self.conf_win_bottom_offset));
        (w, h)
    }

    /// The DPB sizing of the highest sub-layer, which is the one decoded.
    pub fn highest_dec_pic_buf(&self) -> (u32, u32) {
        let top = self.max_sub_layers_minus1 as usize;
        (
            self.dec_pic_buf.max_dec_pic_buffering_minus1[top] as u32 + 1,
            self.dec_pic_buf.max_num_reorder_pics[top] as u32,
        )
    }
}

pub(crate) fn parse_sps(payload: &[u8]) -> Result<Sps> {
    let mut r = BitReader::new(payload);
    let mut sps = Sps {
        vps_id: r.bits(4)? as u8,
        ..Default::default()
    };
    let max_sub_layers_minus1 = r.bits(3)?;
    if max_sub_layers_minus1 > 6 {
        return Err(invalid("sps_max_sub_layers_minus1 out of range"));
    }
    sps.max_sub_layers_minus1 = max_sub_layers_minus1 as u8;
    sps.temporal_id_nesting_flag = r.flag()?;
    sps.ptl = parse_profile_tier_level(&mut r, max_sub_layers_minus1)?;
    let sps_id = r.ue()?;
    if sps_id > 15 {
        return Err(invalid("sps_seq_parameter_set_id out of range"));
    }
    sps.sps_id = sps_id as u8;
    let chroma_format_idc = r.ue()?;
    if chroma_format_idc > 3 {
        return Err(invalid("chroma_format_idc out of range"));
    }
    sps.chroma_format_idc = chroma_format_idc as u8;
    if chroma_format_idc == 3 {
        sps.separate_colour_plane_flag = r.flag()?;
    }
    sps.pic_width_in_luma_samples = r.ue()?;
    sps.pic_height_in_luma_samples = r.ue()?;
    sps.conformance_window_flag = r.flag()?;
    if sps.conformance_window_flag {
        sps.conf_win_left_offset = r.ue()?;
        sps.conf_win_right_offset = r.ue()?;
        sps.conf_win_top_offset = r.ue()?;
        sps.conf_win_bottom_offset = r.ue()?;
    }
    sps.bit_depth_luma_minus8 = r.ue()?.min(255) as u8;
    sps.bit_depth_chroma_minus8 = r.ue()?.min(255) as u8;
    let log2_max_poc_lsb_minus4 = r.ue()?;
    if log2_max_poc_lsb_minus4 > 12 {
        return Err(invalid("log2_max_pic_order_cnt_lsb_minus4 out of range"));
    }
    sps.log2_max_pic_order_cnt_lsb_minus4 = log2_max_poc_lsb_minus4 as u8;
    let (dec_pic_buf, ordering_present) = parse_dec_pic_buf(&mut r, max_sub_layers_minus1)?;
    sps.dec_pic_buf = dec_pic_buf;
    sps.sub_layer_ordering_info_present_flag = ordering_present;
    sps.log2_min_luma_coding_block_size_minus3 = r.ue()?.min(255) as u8;
    sps.log2_diff_max_min_luma_coding_block_size = r.ue()?.min(255) as u8;
    sps.log2_min_luma_transform_block_size_minus2 = r.ue()?.min(255) as u8;
    sps.log2_diff_max_min_luma_transform_block_size = r.ue()?.min(255) as u8;
    sps.max_transform_hierarchy_depth_inter = r.ue()?.min(255) as u8;
    sps.max_transform_hierarchy_depth_intra = r.ue()?.min(255) as u8;
    if sps.ctb_log2_size() > 6 || sps.ctb_log2_size() < 4 {
        return Err(invalid("coding tree block size out of range"));
    }

    sps.scaling_list_enabled_flag = r.flag()?;
    if sps.scaling_list_enabled_flag {
        sps.sps_scaling_list_data_present_flag = r.flag()?;
        sps.scaling_lists = Some(if sps.sps_scaling_list_data_present_flag {
            parse_scaling_list_data(&mut r)?
        } else {
            ScalingLists::defaults()
        });
    }
    sps.amp_enabled_flag = r.flag()?;
    sps.sample_adaptive_offset_enabled_flag = r.flag()?;
    sps.pcm_enabled_flag = r.flag()?;
    if sps.pcm_enabled_flag {
        sps.pcm_sample_bit_depth_luma_minus1 = r.bits(4)? as u8;
        sps.pcm_sample_bit_depth_chroma_minus1 = r.bits(4)? as u8;
        sps.log2_min_pcm_luma_coding_block_size_minus3 = r.ue()?.min(255) as u8;
        sps.log2_diff_max_min_pcm_luma_coding_block_size = r.ue()?.min(255) as u8;
        sps.pcm_loop_filter_disabled_flag = r.flag()?;
    }
    let num_short_term_ref_pic_sets = r.ue()? as usize;
    if num_short_term_ref_pic_sets > 64 {
        return Err(invalid("num_short_term_ref_pic_sets out of range"));
    }
    for i in 0..num_short_term_ref_pic_sets {
        let rps =
            parse_short_term_rps(&mut r, i, num_short_term_ref_pic_sets, &sps.short_term_rps)?;
        sps.short_term_rps.push(rps);
    }
    sps.long_term_ref_pics_present_flag = r.flag()?;
    if sps.long_term_ref_pics_present_flag {
        let num = r.ue()?;
        if num > 32 {
            return Err(invalid("num_long_term_ref_pics_sps out of range"));
        }
        for _ in 0..num {
            sps.lt_ref_pic_poc_lsb_sps
                .push(r.bits(sps.log2_max_pic_order_cnt_lsb())?);
            sps.used_by_curr_pic_lt_sps_flag.push(r.flag()?);
        }
    }
    sps.sps_temporal_mvp_enabled_flag = r.flag()?;
    sps.strong_intra_smoothing_enabled_flag = r.flag()?;
    if r.flag()? {
        skip_vui(&mut r, max_sub_layers_minus1)?;
    }
    if r.flag()? {
        let range_extension = r.flag()?;
        let multilayer_extension = r.flag()?;
        let extension_3d = r.flag()?;
        let scc_extension = r.flag()?;
        r.skip(4)?;
        if range_extension {
            sps.range_extension = SpsRangeExtension {
                present: true,
                transform_skip_rotation_enabled_flag: r.flag()?,
                transform_skip_context_enabled_flag: r.flag()?,
                implicit_rdpcm_enabled_flag: r.flag()?,
                explicit_rdpcm_enabled_flag: r.flag()?,
                extended_precision_processing_flag: r.flag()?,
                intra_smoothing_disabled_flag: r.flag()?,
                high_precision_offsets_enabled_flag: r.flag()?,
                persistent_rice_adaptation_enabled_flag: r.flag()?,
                cabac_bypass_alignment_enabled_flag: r.flag()?,
            };
        }
        // The SCC extension follows the other two, whose layout is not read
        // here, so it can only be refused rather than parsed.
        if scc_extension {
            return Err(invalid("screen content coding streams are not supported"));
        }
        let _ = (multilayer_extension, extension_3d);
    }
    Ok(sps)
}

/// Parsed picture parameter set (the subset Vulkan decode needs).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Pps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub init_qp_minus26: i8,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub uniform_spacing_flag: bool,
    /// Coded widths of every tile column but the last, empty when the spacing
    /// is uniform.
    pub column_width_minus1: Vec<u16>,
    pub row_height_minus1: Vec<u16>,
    pub loop_filter_across_tiles_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub pps_scaling_list_data_present_flag: bool,
    pub scaling_lists: Option<ScalingLists>,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u8,
    pub slice_segment_header_extension_present_flag: bool,
    pub range_extension: PpsRangeExtension,
}

/// The fields of `pps_range_extension()`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PpsRangeExtension {
    pub present: bool,
    pub log2_max_transform_skip_block_size_minus2: u8,
    pub cross_component_prediction_enabled_flag: bool,
    pub chroma_qp_offset_list_enabled_flag: bool,
    pub diff_cu_chroma_qp_offset_depth: u8,
    pub chroma_qp_offset_list_len_minus1: u8,
    pub cb_qp_offset_list: [i8; 6],
    pub cr_qp_offset_list: [i8; 6],
    pub log2_sao_offset_scale_luma: u8,
    pub log2_sao_offset_scale_chroma: u8,
}

pub(crate) fn parse_pps(payload: &[u8]) -> Result<Pps> {
    let mut r = BitReader::new(payload);
    let mut pps = Pps::default();
    let pps_id = r.ue()?;
    let sps_id = r.ue()?;
    if pps_id > 63 || sps_id > 15 {
        return Err(invalid("parameter set id out of range"));
    }
    pps.pps_id = pps_id as u8;
    pps.sps_id = sps_id as u8;
    pps.dependent_slice_segments_enabled_flag = r.flag()?;
    pps.output_flag_present_flag = r.flag()?;
    pps.num_extra_slice_header_bits = r.bits(3)? as u8;
    pps.sign_data_hiding_enabled_flag = r.flag()?;
    pps.cabac_init_present_flag = r.flag()?;
    pps.num_ref_idx_l0_default_active_minus1 = r.ue()?.min(255) as u8;
    pps.num_ref_idx_l1_default_active_minus1 = r.ue()?.min(255) as u8;
    pps.init_qp_minus26 = r.se()? as i8;
    pps.constrained_intra_pred_flag = r.flag()?;
    pps.transform_skip_enabled_flag = r.flag()?;
    pps.cu_qp_delta_enabled_flag = r.flag()?;
    if pps.cu_qp_delta_enabled_flag {
        pps.diff_cu_qp_delta_depth = r.ue()?.min(255) as u8;
    }
    pps.pps_cb_qp_offset = r.se()? as i8;
    pps.pps_cr_qp_offset = r.se()? as i8;
    pps.pps_slice_chroma_qp_offsets_present_flag = r.flag()?;
    pps.weighted_pred_flag = r.flag()?;
    pps.weighted_bipred_flag = r.flag()?;
    pps.transquant_bypass_enabled_flag = r.flag()?;
    pps.tiles_enabled_flag = r.flag()?;
    pps.entropy_coding_sync_enabled_flag = r.flag()?;
    if pps.tiles_enabled_flag {
        let columns_minus1 = r.ue()?;
        let rows_minus1 = r.ue()?;
        if columns_minus1 > 19 || rows_minus1 > 21 {
            return Err(invalid("tile grid out of range"));
        }
        pps.num_tile_columns_minus1 = columns_minus1 as u8;
        pps.num_tile_rows_minus1 = rows_minus1 as u8;
        pps.uniform_spacing_flag = r.flag()?;
        if !pps.uniform_spacing_flag {
            for _ in 0..columns_minus1 {
                pps.column_width_minus1
                    .push(r.ue()?.min(u16::MAX as u32) as u16);
            }
            for _ in 0..rows_minus1 {
                pps.row_height_minus1
                    .push(r.ue()?.min(u16::MAX as u32) as u16);
            }
        }
        pps.loop_filter_across_tiles_enabled_flag = r.flag()?;
    } else {
        pps.uniform_spacing_flag = true;
        pps.loop_filter_across_tiles_enabled_flag = true;
    }
    pps.pps_loop_filter_across_slices_enabled_flag = r.flag()?;
    pps.deblocking_filter_control_present_flag = r.flag()?;
    if pps.deblocking_filter_control_present_flag {
        pps.deblocking_filter_override_enabled_flag = r.flag()?;
        pps.pps_deblocking_filter_disabled_flag = r.flag()?;
        if !pps.pps_deblocking_filter_disabled_flag {
            pps.pps_beta_offset_div2 = r.se()? as i8;
            pps.pps_tc_offset_div2 = r.se()? as i8;
        }
    }
    pps.pps_scaling_list_data_present_flag = r.flag()?;
    if pps.pps_scaling_list_data_present_flag {
        pps.scaling_lists = Some(parse_scaling_list_data(&mut r)?);
    }
    pps.lists_modification_present_flag = r.flag()?;
    pps.log2_parallel_merge_level_minus2 = r.ue()?.min(255) as u8;
    pps.slice_segment_header_extension_present_flag = r.flag()?;
    if r.flag()? {
        let range_extension = r.flag()?;
        let multilayer_extension = r.flag()?;
        let extension_3d = r.flag()?;
        let scc_extension = r.flag()?;
        r.skip(4)?;
        if range_extension {
            let mut ext = PpsRangeExtension {
                present: true,
                ..Default::default()
            };
            if pps.transform_skip_enabled_flag {
                ext.log2_max_transform_skip_block_size_minus2 = r.ue()?.min(255) as u8;
            }
            ext.cross_component_prediction_enabled_flag = r.flag()?;
            ext.chroma_qp_offset_list_enabled_flag = r.flag()?;
            if ext.chroma_qp_offset_list_enabled_flag {
                ext.diff_cu_chroma_qp_offset_depth = r.ue()?.min(255) as u8;
                let len_minus1 = r.ue()?;
                if len_minus1 > 5 {
                    return Err(invalid("chroma_qp_offset_list_len_minus1 out of range"));
                }
                ext.chroma_qp_offset_list_len_minus1 = len_minus1 as u8;
                for i in 0..=len_minus1 as usize {
                    ext.cb_qp_offset_list[i] = r.se()? as i8;
                    ext.cr_qp_offset_list[i] = r.se()? as i8;
                }
            }
            ext.log2_sao_offset_scale_luma = r.ue()?.min(255) as u8;
            ext.log2_sao_offset_scale_chroma = r.ue()?.min(255) as u8;
            pps.range_extension = ext;
        }
        if scc_extension {
            return Err(invalid("screen content coding streams are not supported"));
        }
        let _ = (multilayer_extension, extension_3d);
    }
    Ok(pps)
}

/// The kind of a slice (`slice_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SliceType {
    B,
    P,
    I,
}

/// One entry of the long-term part of a slice header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LongTermRef {
    /// `PocLsbLt`.
    pub poc_lsb: u32,
    pub used_by_curr_pic: bool,
    pub delta_poc_msb_present_flag: bool,
    /// `DeltaPocMsbCycleLt`, already accumulated as the spec defines it.
    pub delta_poc_msb_cycle: u32,
}

/// The leading part of a slice segment header: everything up to the picture
/// order count and reference picture set syntax.
#[derive(Debug, Clone)]
pub(crate) struct SliceHeader {
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    pub pic_order_cnt_lsb: u32,
    /// Whether the short-term RPS is one of the SPS's.
    pub short_term_ref_pic_set_sps_flag: bool,
    /// The short-term RPS in force: the SPS's, or the one coded in the header.
    pub short_term_rps: ShortTermRps,
    /// Bits the header spends on a short-term RPS it codes itself, which the
    /// driver needs to skip over it. Zero when the SPS's is used.
    pub short_term_rps_bits: u32,
    pub long_term_refs: Vec<LongTermRef>,
}

/// Parse the slice segment header of `nal` as far as the reference picture
/// set. A dependent slice segment has no such syntax: it takes everything from
/// the segment before it, so only its leading fields are filled in.
pub(crate) fn parse_slice_header(nal: &NalUnit, sps: &Sps, pps: &Pps) -> Result<SliceHeader> {
    let mut r = BitReader::new(nal.payload());
    let first_slice_segment_in_pic_flag = r.flag()?;
    if nal.is_irap() {
        r.skip(1)?; // no_output_of_prior_pics_flag, which is not honoured
    }
    r.ue()?; // slice_pic_parameter_set_id, found earlier by peeking
    let mut dependent_slice_segment_flag = false;
    if !first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            dependent_slice_segment_flag = r.flag()?;
        }
        r.skip(ceil_log2(sps.pic_size_in_ctbs()))?;
    }
    let mut header = SliceHeader {
        slice_type: SliceType::I,
        pic_output_flag: true,
        pic_order_cnt_lsb: 0,
        short_term_ref_pic_set_sps_flag: false,
        short_term_rps: ShortTermRps::default(),
        short_term_rps_bits: 0,
        long_term_refs: Vec::new(),
    };
    if dependent_slice_segment_flag {
        return Ok(header);
    }

    r.skip(pps.num_extra_slice_header_bits as u32)?;
    header.slice_type = match r.ue()? {
        0 => SliceType::B,
        1 => SliceType::P,
        2 => SliceType::I,
        _ => return Err(invalid("slice_type out of range")),
    };
    if pps.output_flag_present_flag {
        header.pic_output_flag = r.flag()?;
    }
    if sps.separate_colour_plane_flag {
        r.skip(2)?;
    }
    if is_idr(nal.nal_type) {
        return Ok(header);
    }

    header.pic_order_cnt_lsb = r.bits(sps.log2_max_pic_order_cnt_lsb())?;
    header.short_term_ref_pic_set_sps_flag = r.flag()?;
    let num_sets = sps.short_term_rps.len();
    if !header.short_term_ref_pic_set_sps_flag {
        let start = r.position();
        header.short_term_rps =
            parse_short_term_rps(&mut r, num_sets, num_sets, &sps.short_term_rps)?;
        header.short_term_rps_bits = (r.position() - start) as u32;
    } else {
        let idx = if num_sets > 1 {
            r.bits(ceil_log2(num_sets as u32))? as usize
        } else {
            0
        };
        header.short_term_rps = sps
            .short_term_rps
            .get(idx)
            .ok_or_else(|| invalid("short_term_ref_pic_set_idx out of range"))?
            .clone();
    }

    if sps.long_term_ref_pics_present_flag {
        let num_sps_candidates = sps.lt_ref_pic_poc_lsb_sps.len() as u32;
        let num_long_term_sps = if num_sps_candidates > 0 { r.ue()? } else { 0 };
        let num_long_term_pics = r.ue()?;
        if num_long_term_sps > num_sps_candidates || num_long_term_sps + num_long_term_pics > 32 {
            return Err(invalid("long-term picture count out of range"));
        }
        let mut cycle = 0u32;
        for i in 0..num_long_term_sps + num_long_term_pics {
            let (poc_lsb, used_by_curr_pic) = if i < num_long_term_sps {
                let idx = if num_sps_candidates > 1 {
                    r.bits(ceil_log2(num_sps_candidates))? as usize
                } else {
                    0
                };
                if idx >= sps.lt_ref_pic_poc_lsb_sps.len() {
                    return Err(invalid("lt_idx_sps out of range"));
                }
                (
                    sps.lt_ref_pic_poc_lsb_sps[idx],
                    sps.used_by_curr_pic_lt_sps_flag[idx],
                )
            } else {
                (r.bits(sps.log2_max_pic_order_cnt_lsb())?, r.flag()?)
            };
            let delta_poc_msb_present_flag = r.flag()?;
            let delta = if delta_poc_msb_present_flag {
                r.ue()?
            } else {
                0
            };
            // The cycle accumulates within the SPS-coded entries and again
            // within the slice-coded ones, restarting at the boundary.
            if i == 0 || i == num_long_term_sps {
                cycle = delta;
            } else {
                cycle += delta;
            }
            header.long_term_refs.push(LongTermRef {
                poc_lsb,
                used_by_curr_pic,
                delta_poc_msb_present_flag,
                delta_poc_msb_cycle: cycle,
            });
        }
    }
    Ok(header)
}
