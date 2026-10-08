//! The AV1 frame header, and the reference state it is parsed against.
//!
//! Vulkan Video decode takes the frame header as parsed data rather than as
//! bits, so this is a full `uncompressed_header()`, not just the
//! prefix the H.26x parsers need. The syntax elements are named as the
//! specification names them.
//!
//! Parsing a header depends on earlier frames: sizes, order hints, saved loop
//! filter and segmentation state and global motion parameters all come from the
//! reference slots, so the parser keeps those slots as the decoding process
//! would.

use std::sync::Arc;

use crate::decoder::av1::reader::BitReader;
use crate::decoder::av1::sequence::{Obu, SELECT, SequenceHeader};
use crate::error::{PixelForgeError, Result};

pub(crate) const NUM_REF_FRAMES: usize = 8;
pub(crate) const REFS_PER_FRAME: usize = 7;
pub(crate) const TOTAL_REFS_PER_FRAME: usize = 8;
pub(crate) const PRIMARY_REF_NONE: u8 = 7;
pub(crate) const MAX_SEGMENTS: usize = 8;
pub(crate) const SEG_LVL_MAX: usize = 8;
pub(crate) const MAX_TILE_COLS: usize = 64;
pub(crate) const MAX_TILE_ROWS: usize = 64;

const SEG_LVL_ALT_Q: usize = 0;
const SUPERRES_NUM: u32 = 8;
const SUPERRES_DENOM_MIN: u32 = 9;
const SUPERRES_DENOM_BITS: u32 = 3;
const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;
const WARPEDMODEL_PREC_BITS: u32 = 16;
const GM_ABS_ALPHA_BITS: u32 = 12;
const GM_ALPHA_PREC_BITS: u32 = 15;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_TRANS_PREC_BITS: u32 = 6;

/// `Segmentation_Feature_Bits`, `_Max` and `_Signed`.
const SEG_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEG_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];
const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];

/// Default `loop_filter_ref_deltas`.
const DEFAULT_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, -1, 0, -1, -1];

/// `frame_type`.
pub(crate) const KEY_FRAME: u8 = 0;
pub(crate) const INTER_FRAME: u8 = 1;
pub(crate) const INTRA_ONLY_FRAME: u8 = 2;
pub(crate) const SWITCH_FRAME: u8 = 3;

/// Reference frame names, which index the per-reference arrays.
pub(crate) const LAST_FRAME: usize = 1;

/// Why a header could not be parsed.
#[derive(Debug)]
pub(crate) enum HeaderError {
    /// The header names something that is not there: a reference that was
    /// never decoded, or a sequence header that never arrived. What joining a
    /// stream partway, or losing data, looks like. Recoverable with a keyframe.
    MissingState(String),
    /// The stream is malformed or uses something unsupported.
    Invalid(PixelForgeError),
}

impl From<PixelForgeError> for HeaderError {
    fn from(e: PixelForgeError) -> Self {
        HeaderError::Invalid(e)
    }
}

type HResult<T> = std::result::Result<T, HeaderError>;

fn err(what: &str) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("AV1 decode: {what}"))
}

fn invalid(what: &str) -> HeaderError {
    HeaderError::Invalid(err(what))
}

fn missing(what: impl Into<String>) -> HeaderError {
    HeaderError::MissingState(what.into())
}

/// What the decoding process remembers about a frame in a reference slot.
#[derive(Debug, Clone)]
struct RefSlot {
    valid: bool,
    frame_id: u32,
    upscaled_width: u32,
    frame_height: u32,
    render_width: u32,
    render_height: u32,
    frame_type: u8,
    order_hint: u32,
    showable_frame: bool,
    gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    lf_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    lf_mode_deltas: [i8; 2],
    seg_feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    seg_feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

fn default_gm_params() -> [[i32; 6]; TOTAL_REFS_PER_FRAME] {
    let mut p = [[0i32; 6]; TOTAL_REFS_PER_FRAME];
    for row in p.iter_mut() {
        row[2] = 1 << WARPEDMODEL_PREC_BITS;
        row[5] = 1 << WARPEDMODEL_PREC_BITS;
    }
    p
}

impl Default for RefSlot {
    fn default() -> Self {
        Self {
            valid: false,
            frame_id: 0,
            upscaled_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            frame_type: KEY_FRAME,
            order_hint: 0,
            showable_frame: false,
            gm_params: default_gm_params(),
            lf_ref_deltas: DEFAULT_REF_DELTAS,
            lf_mode_deltas: [0; 2],
            seg_feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            seg_feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }
}

/// `tile_info()` and what is derived from it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TileInfo {
    pub uniform_tile_spacing_flag: bool,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub context_update_tile_id: u32,
    pub tile_size_bytes: u32,
    /// `MiColStarts`, with `tile_cols + 1` entries used.
    pub mi_col_starts: [u32; MAX_TILE_COLS + 1],
    pub mi_row_starts: [u32; MAX_TILE_ROWS + 1],
    pub width_in_sbs_minus_1: [u32; MAX_TILE_COLS],
    pub height_in_sbs_minus_1: [u32; MAX_TILE_ROWS],
}

impl Default for TileInfo {
    fn default() -> Self {
        Self {
            uniform_tile_spacing_flag: false,
            tile_cols: 1,
            tile_rows: 1,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            context_update_tile_id: 0,
            tile_size_bytes: 0,
            mi_col_starts: [0; MAX_TILE_COLS + 1],
            mi_row_starts: [0; MAX_TILE_ROWS + 1],
            width_in_sbs_minus_1: [0; MAX_TILE_COLS],
            height_in_sbs_minus_1: [0; MAX_TILE_ROWS],
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Quantization {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub diff_uv_delta: bool,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LoopFilter {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub mode_deltas: [i8; 2],
}

/// `cdef_params()`.
///
/// The secondary strengths are kept as the syntax codes them, `0..=3`. The
/// specification's own variable steps a coded 3 up to 4, but the `StdVideo`
/// struct takes the coded value and the driver does that step itself, so
/// passing 4 decodes wrongly, and only for the streams whose encoder picks
/// that strength.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Cdef {
    pub damping_minus_3: u8,
    pub bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

/// `FrameRestorationType` as the syntax codes it after `Remap_Lr_Type`.
pub(crate) const RESTORE_NONE: u8 = 0;
pub(crate) const RESTORE_WIENER: u8 = 1;
pub(crate) const RESTORE_SGRPROJ: u8 = 2;
pub(crate) const RESTORE_SWITCHABLE: u8 = 3;

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LoopRestoration {
    pub frame_restoration_type: [u8; 3],
    /// `log2(LoopRestorationSize) - 5`, which is how Vulkan takes it.
    pub size_log2_minus_5: [u8; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GlobalMotion {
    /// `GmType` by reference name: identity, translation, rot-zoom, affine.
    pub gm_type: [u8; TOTAL_REFS_PER_FRAME],
    pub gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
}

impl Default for GlobalMotion {
    fn default() -> Self {
        Self {
            gm_type: [0; TOTAL_REFS_PER_FRAME],
            gm_params: default_gm_params(),
        }
    }
}

/// A parsed `uncompressed_header()`.
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: u8,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub buffer_removal_time_present_flag: bool,
    pub refresh_frame_flags: u8,
    pub allow_intrabc: bool,
    pub frame_refs_short_signaling: bool,
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    /// `interpolation_filter`: 0 to 3, or 4 for switchable.
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub render_and_frame_size_different: bool,
    pub use_superres: bool,
    pub coded_denom: u8,
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    /// `OrderHints` by reference name.
    pub order_hints: [u32; TOTAL_REFS_PER_FRAME],
    pub ref_frame_sign_bias: [bool; TOTAL_REFS_PER_FRAME],
    pub tile_info: TileInfo,
    pub quant: Quantization,
    pub segmentation: Segmentation,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    pub loop_filter: LoopFilter,
    pub cdef: Cdef,
    pub loop_restoration: LoopRestoration,
    /// `TxMode`: 0 only 4x4, 1 largest, 2 select.
    pub tx_mode: u8,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub skip_mode_frame: [u8; 2],
    pub global_motion: GlobalMotion,
    /// `apply_grain`. The grain parameters themselves are parsed past but not
    /// kept: film grain is not applied.
    pub apply_grain: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    /// Bytes the header took, rounded up to a whole byte. For an `OBU_FRAME`
    /// the tile group starts there.
    pub header_bytes: usize,
}

impl FrameHeader {
    pub fn frame_is_intra(&self) -> bool {
        matches!(self.frame_type, KEY_FRAME | INTRA_ONLY_FRAME)
    }
}

/// One tile of a tile group: where its data sits in the OBU payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TileData {
    pub offset: usize,
    pub size: usize,
}

/// A parsed `tile_group_obu()`.
#[derive(Debug, Clone)]
pub(crate) struct TileGroup {
    pub tg_end: u32,
    pub tiles: Vec<TileData>,
}

pub(crate) struct HeaderParser {
    pub seq: Option<Arc<SequenceHeader>>,
    refs: [RefSlot; NUM_REF_FRAMES],
    current_frame_id: u32,
}

/// `get_relative_dist`.
fn relative_dist(seq: &SequenceHeader, a: u32, b: u32) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1i32 << (seq.order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

impl HeaderParser {
    pub fn new() -> Self {
        Self {
            seq: None,
            refs: Default::default(),
            current_frame_id: 0,
        }
    }

    /// Forget every reference: after a new sequence header changes what the
    /// frames in them were coded as.
    pub fn reset_references(&mut self) {
        self.refs = Default::default();
    }

    fn seq(&self) -> HResult<Arc<SequenceHeader>> {
        self.seq
            .clone()
            .ok_or_else(|| missing("no sequence header yet"))
    }

    /// Parse the frame header at the start of `payload`, then update the
    /// reference state the way decoding the frame would, so the
    /// next header sees this frame.
    pub fn parse(&mut self, obu: &Obu, payload: &[u8]) -> HResult<FrameHeader> {
        let seq = self.seq()?;
        let mut r = BitReader::new(payload);
        let mut fh = FrameHeader::default();
        let id_len = if seq.frame_id_numbers_present_flag {
            seq.frame_id_length()
        } else {
            0
        };
        const ALL_FRAMES: u8 = 0xFF;

        if seq.reduced_still_picture_header {
            fh.frame_type = KEY_FRAME;
            fh.show_frame = true;
        } else {
            fh.show_existing_frame = r.flag()?;
            if fh.show_existing_frame {
                fh.frame_to_show_map_idx = r.bits(3)? as u8;
                if let Some(model) = &seq.decoder_model_info
                    && !seq.timing_info.is_some_and(|t| t.equal_picture_interval)
                {
                    r.skip(model.frame_presentation_time_length_minus_1 as usize + 1)?;
                }
                if seq.frame_id_numbers_present_flag {
                    r.skip(id_len as usize)?;
                }
                let slot = &self.refs[fh.frame_to_show_map_idx as usize];
                if !slot.valid {
                    return Err(missing("show_existing_frame names an empty slot"));
                }
                if !slot.showable_frame {
                    return Err(invalid(
                        "show_existing_frame shows a frame that is not showable",
                    ));
                }
                fh.frame_type = slot.frame_type;
                fh.show_frame = true;
                fh.refresh_frame_flags = if fh.frame_type == KEY_FRAME {
                    ALL_FRAMES
                } else {
                    0
                };
                fh.upscaled_width = slot.upscaled_width;
                fh.frame_height = slot.frame_height;
                fh.current_frame_id = slot.frame_id;
                fh.order_hint = slot.order_hint;
                if obu.obu_type == crate::decoder::av1::sequence::obu_type::FRAME {
                    return Err(invalid("an OBU_FRAME cannot show an existing frame"));
                }
                // Showing a key frame this way refreshes every reference slot with it.
                if fh.frame_type == KEY_FRAME {
                    let shown = self.refs[fh.frame_to_show_map_idx as usize].clone();
                    for slot in self.refs.iter_mut() {
                        *slot = shown.clone();
                    }
                    self.current_frame_id = shown.frame_id;
                }
                fh.header_bytes = r.position().div_ceil(8);
                return Ok(fh);
            }
            fh.frame_type = r.bits(2)? as u8;
            fh.show_frame = r.flag()?;
            if fh.show_frame
                && let Some(model) = &seq.decoder_model_info
                && !seq.timing_info.is_some_and(|t| t.equal_picture_interval)
            {
                r.skip(model.frame_presentation_time_length_minus_1 as usize + 1)?;
            }
            fh.showable_frame = if fh.show_frame {
                fh.frame_type != KEY_FRAME
            } else {
                r.flag()?
            };
            fh.error_resilient_mode =
                if fh.frame_type == SWITCH_FRAME || (fh.frame_type == KEY_FRAME && fh.show_frame) {
                    true
                } else {
                    r.flag()?
                };
        }

        let frame_is_intra = fh.frame_is_intra();
        if fh.frame_type == KEY_FRAME && fh.show_frame {
            for slot in self.refs.iter_mut() {
                slot.valid = false;
                slot.order_hint = 0;
            }
        }

        fh.disable_cdf_update = r.flag()?;
        fh.allow_screen_content_tools = if seq.seq_force_screen_content_tools == SELECT {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        fh.force_integer_mv = if fh.allow_screen_content_tools {
            if seq.seq_force_integer_mv == SELECT {
                r.flag()?
            } else {
                seq.seq_force_integer_mv != 0
            }
        } else {
            false
        };
        if frame_is_intra {
            fh.force_integer_mv = true;
        }

        if seq.frame_id_numbers_present_flag {
            self.current_frame_id = r.bits(id_len)?;
        } else {
            self.current_frame_id = 0;
        }
        fh.current_frame_id = self.current_frame_id;

        fh.frame_size_override_flag = if fh.frame_type == SWITCH_FRAME {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };

        fh.order_hint = r.bits(seq.order_hint_bits)?;
        fh.primary_ref_frame = if frame_is_intra || fh.error_resilient_mode {
            PRIMARY_REF_NONE
        } else {
            r.bits(3)? as u8
        };

        if let Some(model) = &seq.decoder_model_info {
            fh.buffer_removal_time_present_flag = r.flag()?;
            if fh.buffer_removal_time_present_flag {
                for op in &seq.operating_points {
                    if op.decoder_model_present_for_this_op {
                        let in_temporal = (op.idc >> obu.temporal_id) & 1 != 0;
                        let in_spatial = (op.idc >> (obu.spatial_id + 8)) & 1 != 0;
                        if op.idc == 0 || (in_temporal && in_spatial) {
                            r.skip(model.buffer_removal_time_length_minus_1 as usize + 1)?;
                        }
                    }
                }
            }
        }

        fh.refresh_frame_flags =
            if fh.frame_type == SWITCH_FRAME || (fh.frame_type == KEY_FRAME && fh.show_frame) {
                ALL_FRAMES
            } else {
                r.bits(8)? as u8
            };
        if frame_is_intra && fh.frame_type == INTRA_ONLY_FRAME && fh.refresh_frame_flags == 0xFF {
            return Err(invalid("an intra-only frame cannot refresh every slot"));
        }

        if (!frame_is_intra || fh.refresh_frame_flags != ALL_FRAMES)
            && fh.error_resilient_mode
            && seq.enable_order_hint
        {
            for i in 0..NUM_REF_FRAMES {
                let ref_order_hint = r.bits(seq.order_hint_bits)?;
                if ref_order_hint != self.refs[i].order_hint {
                    self.refs[i].valid = false;
                    self.refs[i].order_hint = ref_order_hint;
                }
            }
        }

        if frame_is_intra {
            self.frame_size(&seq, &mut fh, &mut r)?;
            self.render_size(&mut fh, &mut r)?;
            if fh.allow_screen_content_tools && fh.upscaled_width == fh.frame_width {
                fh.allow_intrabc = r.flag()?;
            }
        } else {
            if seq.enable_order_hint {
                fh.frame_refs_short_signaling = r.flag()?;
                if fh.frame_refs_short_signaling {
                    let last_frame_idx = r.bits(3)? as u8;
                    let gold_frame_idx = r.bits(3)? as u8;
                    fh.ref_frame_idx =
                        self.set_frame_refs(&seq, fh.order_hint, last_frame_idx, gold_frame_idx);
                }
            }
            for i in 0..REFS_PER_FRAME {
                if !fh.frame_refs_short_signaling {
                    fh.ref_frame_idx[i] = r.bits(3)? as u8;
                }
                if seq.frame_id_numbers_present_flag {
                    r.skip(seq.delta_frame_id_length_minus_2 as usize + 2)?;
                }
            }
            for &idx in &fh.ref_frame_idx {
                if !self.refs[idx as usize].valid {
                    return Err(missing(format!("reference slot {idx} is empty")));
                }
            }
            if fh.frame_size_override_flag && !fh.error_resilient_mode {
                self.frame_size_with_refs(&seq, &mut fh, &mut r)?;
            } else {
                self.frame_size(&seq, &mut fh, &mut r)?;
                self.render_size(&mut fh, &mut r)?;
            }
            fh.allow_high_precision_mv = if fh.force_integer_mv {
                false
            } else {
                r.flag()?
            };
            fh.is_filter_switchable = r.flag()?;
            fh.interpolation_filter = if fh.is_filter_switchable {
                4
            } else {
                r.bits(2)? as u8
            };
            fh.is_motion_mode_switchable = r.flag()?;
            fh.use_ref_frame_mvs = if fh.error_resilient_mode || !seq.enable_ref_frame_mvs {
                false
            } else {
                r.flag()?
            };
            for i in 0..REFS_PER_FRAME {
                let ref_frame = LAST_FRAME + i;
                let hint = self.refs[fh.ref_frame_idx[i] as usize].order_hint;
                fh.order_hints[ref_frame] = hint;
                fh.ref_frame_sign_bias[ref_frame] =
                    seq.enable_order_hint && relative_dist(&seq, hint, fh.order_hint) > 0;
            }
        }

        fh.disable_frame_end_update_cdf =
            if seq.reduced_still_picture_header || fh.disable_cdf_update {
                true
            } else {
                r.flag()?
            };

        // setup_past_independence() or load_previous().
        let mut prev_gm_params = default_gm_params();
        fh.loop_filter.ref_deltas = DEFAULT_REF_DELTAS;
        if fh.primary_ref_frame != PRIMARY_REF_NONE {
            let prev = &self.refs[fh.ref_frame_idx[fh.primary_ref_frame as usize] as usize];
            if !prev.valid {
                return Err(missing("primary reference frame is empty"));
            }
            prev_gm_params = prev.gm_params;
            fh.loop_filter.ref_deltas = prev.lf_ref_deltas;
            fh.loop_filter.mode_deltas = prev.lf_mode_deltas;
            fh.segmentation.feature_enabled = prev.seg_feature_enabled;
            fh.segmentation.feature_data = prev.seg_feature_data;
        }

        self.tile_info(&seq, &mut fh, &mut r)?;
        Self::quantization_params(&seq, &mut fh, &mut r)?;
        self.segmentation_params(&mut fh, &mut r)?;

        // delta_q_params()
        if fh.quant.base_q_idx > 0 {
            fh.delta_q_present = r.flag()?;
        }
        if fh.delta_q_present {
            fh.delta_q_res = r.bits(2)? as u8;
        }
        // delta_lf_params()
        if fh.delta_q_present {
            if !fh.allow_intrabc {
                fh.delta_lf_present = r.flag()?;
            }
            if fh.delta_lf_present {
                fh.delta_lf_res = r.bits(2)? as u8;
                fh.delta_lf_multi = r.flag()?;
            }
        }

        fh.coded_lossless = true;
        for segment_id in 0..MAX_SEGMENTS {
            let q = &fh.quant;
            let qindex = if fh.segmentation.enabled
                && fh.segmentation.feature_enabled[segment_id][SEG_LVL_ALT_Q]
            {
                (q.base_q_idx as i32
                    + fh.segmentation.feature_data[segment_id][SEG_LVL_ALT_Q] as i32)
                    .clamp(0, 255)
            } else {
                q.base_q_idx as i32
            };
            let lossless = qindex == 0
                && q.delta_q_y_dc == 0
                && q.delta_q_u_ac == 0
                && q.delta_q_u_dc == 0
                && q.delta_q_v_ac == 0
                && q.delta_q_v_dc == 0;
            if !lossless {
                fh.coded_lossless = false;
            }
        }
        fh.all_lossless = fh.coded_lossless && fh.frame_width == fh.upscaled_width;

        Self::loop_filter_params(&seq, &mut fh, &mut r)?;
        Self::cdef_params(&seq, &mut fh, &mut r)?;
        Self::lr_params(&seq, &mut fh, &mut r)?;

        // read_tx_mode()
        fh.tx_mode = if fh.coded_lossless {
            0
        } else if r.flag()? {
            2
        } else {
            1
        };
        // frame_reference_mode()
        fh.reference_select = if frame_is_intra { false } else { r.flag()? };
        self.skip_mode_params(&seq, &mut fh, &mut r)?;

        fh.allow_warped_motion =
            if frame_is_intra || fh.error_resilient_mode || !seq.enable_warped_motion {
                false
            } else {
                r.flag()?
            };
        fh.reduced_tx_set = r.flag()?;
        Self::global_motion_params(&mut fh, &mut r, &prev_gm_params)?;
        Self::film_grain_params(&seq, &mut fh, &mut r)?;

        // OBU_FRAME: byte_alignment() before the tile group.
        fh.header_bytes = r.position().div_ceil(8);
        self.update_references(&fh);
        Ok(fh)
    }

    /// Save the frame into every reference slot it refreshes.
    fn update_references(&mut self, fh: &FrameHeader) {
        for (i, slot) in self.refs.iter_mut().enumerate() {
            if (fh.refresh_frame_flags >> i) & 1 == 0 {
                continue;
            }
            *slot = RefSlot {
                valid: true,
                frame_id: fh.current_frame_id,
                upscaled_width: fh.upscaled_width,
                frame_height: fh.frame_height,
                render_width: fh.render_width,
                render_height: fh.render_height,
                frame_type: fh.frame_type,
                order_hint: fh.order_hint,
                showable_frame: fh.showable_frame,
                gm_params: fh.global_motion.gm_params,
                lf_ref_deltas: fh.loop_filter.ref_deltas,
                lf_mode_deltas: fh.loop_filter.mode_deltas,
                seg_feature_enabled: fh.segmentation.feature_enabled,
                seg_feature_data: fh.segmentation.feature_data,
            };
        }
    }

    fn superres_params(
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> Result<()> {
        fh.use_superres = if seq.enable_superres {
            r.flag()?
        } else {
            false
        };
        let denom = if fh.use_superres {
            fh.coded_denom = r.bits(SUPERRES_DENOM_BITS)? as u8;
            fh.coded_denom as u32 + SUPERRES_DENOM_MIN
        } else {
            SUPERRES_NUM
        };
        fh.upscaled_width = fh.frame_width;
        fh.frame_width = (fh.upscaled_width * SUPERRES_NUM + denom / 2) / denom;
        Ok(())
    }

    fn compute_image_size(fh: &mut FrameHeader) {
        fh.mi_cols = 2 * fh.frame_width.div_ceil(8);
        fh.mi_rows = 2 * fh.frame_height.div_ceil(8);
    }

    fn frame_size(
        &self,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> HResult<()> {
        if fh.frame_size_override_flag {
            fh.frame_width = r.bits(seq.frame_width_bits_minus_1 as u32 + 1)? + 1;
            fh.frame_height = r.bits(seq.frame_height_bits_minus_1 as u32 + 1)? + 1;
        } else {
            fh.frame_width = seq.max_frame_width_minus_1 as u32 + 1;
            fh.frame_height = seq.max_frame_height_minus_1 as u32 + 1;
        }
        Self::superres_params(seq, fh, r)?;
        Self::compute_image_size(fh);
        Ok(())
    }

    fn render_size(&self, fh: &mut FrameHeader, r: &mut BitReader) -> HResult<()> {
        fh.render_and_frame_size_different = r.flag()?;
        if fh.render_and_frame_size_different {
            fh.render_width = r.bits(16)? + 1;
            fh.render_height = r.bits(16)? + 1;
        } else {
            fh.render_width = fh.upscaled_width;
            fh.render_height = fh.frame_height;
        }
        Ok(())
    }

    fn frame_size_with_refs(
        &self,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> HResult<()> {
        for i in 0..REFS_PER_FRAME {
            if r.flag()? {
                let slot = &self.refs[fh.ref_frame_idx[i] as usize];
                fh.upscaled_width = slot.upscaled_width;
                fh.frame_width = fh.upscaled_width;
                fh.frame_height = slot.frame_height;
                fh.render_width = slot.render_width;
                fh.render_height = slot.render_height;
                Self::superres_params(seq, fh, r)?;
                Self::compute_image_size(fh);
                return Ok(());
            }
        }
        self.frame_size(seq, fh, r)?;
        self.render_size(fh, r)
    }

    /// Fill in the references a frame does not name, from the two it does: the
    /// closest pictures before and after it, by order hint.
    fn set_frame_refs(
        &self,
        seq: &SequenceHeader,
        order_hint: u32,
        last_frame_idx: u8,
        gold_frame_idx: u8,
    ) -> [u8; REFS_PER_FRAME] {
        let mut ref_frame_idx = [-1i32; REFS_PER_FRAME];
        ref_frame_idx[0] = last_frame_idx as i32;
        ref_frame_idx[4 - 1] = gold_frame_idx as i32;
        let mut used = [false; NUM_REF_FRAMES];
        used[last_frame_idx as usize] = true;
        used[gold_frame_idx as usize] = true;

        let cur_frame_hint = 1i32 << (seq.order_hint_bits - 1);
        let mut shifted = [0i32; NUM_REF_FRAMES];
        for (i, s) in shifted.iter_mut().enumerate() {
            *s = cur_frame_hint + relative_dist(seq, self.refs[i].order_hint, order_hint);
        }
        let mut latest = shifted[last_frame_idx as usize];
        let mut earliest = shifted[gold_frame_idx as usize];

        let find = |used: &[bool; NUM_REF_FRAMES],
                    shifted: &[i32; NUM_REF_FRAMES],
                    backward: bool,
                    prefer_latest: bool,
                    best: &mut i32|
         -> i32 {
            let mut found = -1i32;
            for i in 0..NUM_REF_FRAMES {
                let hint = shifted[i];
                let side = if backward {
                    hint >= cur_frame_hint
                } else {
                    hint < cur_frame_hint
                };
                let better = found < 0
                    || if prefer_latest {
                        hint >= *best
                    } else {
                        hint < *best
                    };
                if !used[i] && side && better {
                    found = i as i32;
                    *best = hint;
                }
            }
            found
        };

        // ALTREF: the latest backward reference.
        let r = find(&used, &shifted, true, true, &mut latest);
        if r >= 0 {
            ref_frame_idx[7 - 1] = r;
            used[r as usize] = true;
        }
        // BWDREF, then ALTREF2: the earliest backward references.
        let r = find(&used, &shifted, true, false, &mut earliest);
        if r >= 0 {
            ref_frame_idx[5 - 1] = r;
            used[r as usize] = true;
        }
        let r = find(&used, &shifted, true, false, &mut earliest);
        if r >= 0 {
            ref_frame_idx[6 - 1] = r;
            used[r as usize] = true;
        }
        // The rest, in this order, take the latest forward references.
        for ref_frame in [2usize, 3, 5, 6, 7] {
            if ref_frame_idx[ref_frame - 1] < 0 {
                let r = find(&used, &shifted, false, true, &mut latest);
                if r >= 0 {
                    ref_frame_idx[ref_frame - 1] = r;
                    used[r as usize] = true;
                }
            }
        }
        // Anything still unset takes the earliest reference there is.
        let mut fallback = 0usize;
        let mut earliest_hint = shifted[0];
        for (i, &hint) in shifted.iter().enumerate().skip(1) {
            if hint < earliest_hint {
                fallback = i;
                earliest_hint = hint;
            }
        }
        let mut out = [0u8; REFS_PER_FRAME];
        for (o, &idx) in out.iter_mut().zip(&ref_frame_idx) {
            *o = if idx < 0 { fallback as u8 } else { idx as u8 };
        }
        out
    }

    fn tile_info(
        &self,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> HResult<()> {
        let (mi_cols, mi_rows) = (fh.mi_cols, fh.mi_rows);
        let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
        let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
        let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
        let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS as u32));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS as u32));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

        let t = &mut fh.tile_info;
        t.uniform_tile_spacing_flag = r.flag()?;
        if t.uniform_tile_spacing_flag {
            t.tile_cols_log2 = min_log2_tile_cols;
            while t.tile_cols_log2 < max_log2_tile_cols && r.flag()? {
                t.tile_cols_log2 += 1;
            }
            let tile_width_sb = (sb_cols + (1 << t.tile_cols_log2) - 1) >> t.tile_cols_log2;
            let mut i = 0usize;
            let mut start_sb = 0;
            while start_sb < sb_cols {
                t.mi_col_starts[i] = start_sb << sb_shift;
                i += 1;
                start_sb += tile_width_sb;
            }
            t.mi_col_starts[i] = mi_cols;
            t.tile_cols = i as u32;

            let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.tile_cols_log2);
            t.tile_rows_log2 = min_log2_tile_rows;
            while t.tile_rows_log2 < max_log2_tile_rows && r.flag()? {
                t.tile_rows_log2 += 1;
            }
            let tile_height_sb = (sb_rows + (1 << t.tile_rows_log2) - 1) >> t.tile_rows_log2;
            let mut i = 0usize;
            let mut start_sb = 0;
            while start_sb < sb_rows {
                t.mi_row_starts[i] = start_sb << sb_shift;
                i += 1;
                start_sb += tile_height_sb;
            }
            t.mi_row_starts[i] = mi_rows;
            t.tile_rows = i as u32;
        } else {
            let mut widest_tile_sb = 0;
            let mut start_sb = 0;
            let mut i = 0usize;
            while start_sb < sb_cols {
                if i >= MAX_TILE_COLS {
                    return Err(invalid("more than 64 tile columns"));
                }
                t.mi_col_starts[i] = start_sb << sb_shift;
                let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
                t.width_in_sbs_minus_1[i] = r.ns(max_width)?;
                let size_sb = t.width_in_sbs_minus_1[i] + 1;
                widest_tile_sb = widest_tile_sb.max(size_sb);
                start_sb += size_sb;
                i += 1;
            }
            t.mi_col_starts[i] = mi_cols;
            t.tile_cols = i as u32;
            t.tile_cols_log2 = tile_log2(1, t.tile_cols);

            max_tile_area_sb = if min_log2_tiles > 0 {
                (sb_rows * sb_cols) >> (min_log2_tiles + 1)
            } else {
                sb_rows * sb_cols
            };
            let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
            let mut start_sb = 0;
            let mut i = 0usize;
            while start_sb < sb_rows {
                if i >= MAX_TILE_ROWS {
                    return Err(invalid("more than 64 tile rows"));
                }
                t.mi_row_starts[i] = start_sb << sb_shift;
                let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
                t.height_in_sbs_minus_1[i] = r.ns(max_height)?;
                start_sb += t.height_in_sbs_minus_1[i] + 1;
                i += 1;
            }
            t.mi_row_starts[i] = mi_rows;
            t.tile_rows = i as u32;
            t.tile_rows_log2 = tile_log2(1, t.tile_rows);
        }
        if t.tile_cols as usize > MAX_TILE_COLS || t.tile_rows as usize > MAX_TILE_ROWS {
            return Err(invalid("too many tiles"));
        }
        // The widths and heights the syntax did not code, for the driver.
        if t.uniform_tile_spacing_flag {
            for i in 0..t.tile_cols as usize {
                t.width_in_sbs_minus_1[i] =
                    ((t.mi_col_starts[i + 1] - t.mi_col_starts[i] + (1 << sb_shift) - 1)
                        >> sb_shift)
                        - 1;
            }
            for i in 0..t.tile_rows as usize {
                t.height_in_sbs_minus_1[i] =
                    ((t.mi_row_starts[i + 1] - t.mi_row_starts[i] + (1 << sb_shift) - 1)
                        >> sb_shift)
                        - 1;
            }
        }
        if t.tile_cols_log2 > 0 || t.tile_rows_log2 > 0 {
            t.context_update_tile_id = r.bits(t.tile_rows_log2 + t.tile_cols_log2)?;
            t.tile_size_bytes = r.bits(2)? + 1;
            if t.context_update_tile_id >= t.tile_rows * t.tile_cols {
                return Err(invalid("context_update_tile_id out of range"));
            }
        }
        Ok(())
    }

    fn delta_q(r: &mut BitReader) -> Result<i8> {
        if r.flag()? { Ok(r.su(7)? as i8) } else { Ok(0) }
    }

    fn quantization_params(
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> Result<()> {
        let q = &mut fh.quant;
        q.base_q_idx = r.bits(8)? as u8;
        q.delta_q_y_dc = Self::delta_q(r)?;
        if seq.num_planes() > 1 {
            q.diff_uv_delta = if seq.color.separate_uv_delta_q {
                r.flag()?
            } else {
                false
            };
            q.delta_q_u_dc = Self::delta_q(r)?;
            q.delta_q_u_ac = Self::delta_q(r)?;
            if q.diff_uv_delta {
                q.delta_q_v_dc = Self::delta_q(r)?;
                q.delta_q_v_ac = Self::delta_q(r)?;
            } else {
                q.delta_q_v_dc = q.delta_q_u_dc;
                q.delta_q_v_ac = q.delta_q_u_ac;
            }
        }
        q.using_qmatrix = r.flag()?;
        if q.using_qmatrix {
            q.qm_y = r.bits(4)? as u8;
            q.qm_u = r.bits(4)? as u8;
            q.qm_v = if seq.color.separate_uv_delta_q {
                r.bits(4)? as u8
            } else {
                q.qm_u
            };
        }
        Ok(())
    }

    fn segmentation_params(&self, fh: &mut FrameHeader, r: &mut BitReader) -> Result<()> {
        let s = &mut fh.segmentation;
        s.enabled = r.flag()?;
        if !s.enabled {
            s.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
            s.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
            return Ok(());
        }
        if fh.primary_ref_frame == PRIMARY_REF_NONE {
            s.update_map = true;
            s.temporal_update = false;
            s.update_data = true;
        } else {
            s.update_map = r.flag()?;
            if s.update_map {
                s.temporal_update = r.flag()?;
            }
            s.update_data = r.flag()?;
        }
        if s.update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = r.flag()?;
                    s.feature_enabled[i][j] = enabled;
                    let mut clipped = 0i32;
                    if enabled {
                        let bits = SEG_FEATURE_BITS[j];
                        let limit = SEG_FEATURE_MAX[j];
                        if SEG_FEATURE_SIGNED[j] {
                            clipped = r.su(1 + bits)?.clamp(-limit, limit);
                        } else {
                            clipped = (r.bits(bits)? as i32).clamp(0, limit);
                        }
                    }
                    s.feature_data[i][j] = clipped as i16;
                }
            }
        }
        Ok(())
    }

    fn loop_filter_params(
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> Result<()> {
        let lf = &mut fh.loop_filter;
        if fh.coded_lossless || fh.allow_intrabc {
            lf.level[0] = 0;
            lf.level[1] = 0;
            lf.ref_deltas = DEFAULT_REF_DELTAS;
            lf.mode_deltas = [0; 2];
            return Ok(());
        }
        lf.level[0] = r.bits(6)? as u8;
        lf.level[1] = r.bits(6)? as u8;
        if seq.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
            lf.level[2] = r.bits(6)? as u8;
            lf.level[3] = r.bits(6)? as u8;
        }
        lf.sharpness = r.bits(3)? as u8;
        lf.delta_enabled = r.flag()?;
        if lf.delta_enabled {
            lf.delta_update = r.flag()?;
            if lf.delta_update {
                for i in 0..TOTAL_REFS_PER_FRAME {
                    if r.flag()? {
                        lf.ref_deltas[i] = r.su(7)? as i8;
                    }
                }
                for i in 0..2 {
                    if r.flag()? {
                        lf.mode_deltas[i] = r.su(7)? as i8;
                    }
                }
            }
        }
        Ok(())
    }

    fn cdef_params(seq: &SequenceHeader, fh: &mut FrameHeader, r: &mut BitReader) -> Result<()> {
        let c = &mut fh.cdef;
        if fh.coded_lossless || fh.allow_intrabc || !seq.enable_cdef {
            c.bits = 0;
            c.damping_minus_3 = 0;
            return Ok(());
        }
        c.damping_minus_3 = r.bits(2)? as u8;
        c.bits = r.bits(2)? as u8;
        for i in 0..(1usize << c.bits) {
            c.y_pri_strength[i] = r.bits(4)? as u8;
            c.y_sec_strength[i] = r.bits(2)? as u8;
            if seq.num_planes() > 1 {
                c.uv_pri_strength[i] = r.bits(4)? as u8;
                c.uv_sec_strength[i] = r.bits(2)? as u8;
            }
        }
        Ok(())
    }

    fn lr_params(seq: &SequenceHeader, fh: &mut FrameHeader, r: &mut BitReader) -> Result<()> {
        let lr = &mut fh.loop_restoration;
        if fh.all_lossless || fh.allow_intrabc || !seq.enable_restoration {
            return Ok(());
        }
        // Remap_Lr_Type.
        const REMAP: [u8; 4] = [
            RESTORE_NONE,
            RESTORE_SWITCHABLE,
            RESTORE_WIENER,
            RESTORE_SGRPROJ,
        ];
        for i in 0..seq.num_planes() as usize {
            lr.frame_restoration_type[i] = REMAP[r.bits(2)? as usize];
            if lr.frame_restoration_type[i] != RESTORE_NONE {
                lr.uses_lr = true;
                if i > 0 {
                    lr.uses_chroma_lr = true;
                }
            }
        }
        if lr.uses_lr {
            let lr_unit_shift = if seq.use_128x128_superblock {
                r.bits(1)? + 1
            } else {
                let mut shift = r.bits(1)?;
                if shift != 0 {
                    shift += r.bits(1)?;
                }
                shift
            };
            let uv_shift =
                if seq.color.subsampling_x && seq.color.subsampling_y && lr.uses_chroma_lr {
                    r.bits(1)?
                } else {
                    0
                };
            lr.size_log2_minus_5[0] = 1 + lr_unit_shift as u8;
            lr.size_log2_minus_5[1] = 1 + lr_unit_shift as u8 - uv_shift as u8;
            lr.size_log2_minus_5[2] = lr.size_log2_minus_5[1];
        }
        Ok(())
    }

    fn skip_mode_params(
        &self,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> Result<()> {
        let allowed = if fh.frame_is_intra() || !fh.reference_select || !seq.enable_order_hint {
            false
        } else {
            let mut forward_idx = -1i32;
            let mut backward_idx = -1i32;
            let mut forward_hint = 0u32;
            let mut backward_hint = 0u32;
            for i in 0..REFS_PER_FRAME {
                let ref_hint = self.refs[fh.ref_frame_idx[i] as usize].order_hint;
                if relative_dist(seq, ref_hint, fh.order_hint) < 0 {
                    if forward_idx < 0 || relative_dist(seq, ref_hint, forward_hint) > 0 {
                        forward_idx = i as i32;
                        forward_hint = ref_hint;
                    }
                } else if relative_dist(seq, ref_hint, fh.order_hint) > 0
                    && (backward_idx < 0 || relative_dist(seq, ref_hint, backward_hint) < 0)
                {
                    backward_idx = i as i32;
                    backward_hint = ref_hint;
                }
            }
            if forward_idx < 0 {
                false
            } else if backward_idx >= 0 {
                fh.skip_mode_frame = [
                    (LAST_FRAME as i32 + forward_idx.min(backward_idx)) as u8,
                    (LAST_FRAME as i32 + forward_idx.max(backward_idx)) as u8,
                ];
                true
            } else {
                let mut second_forward_idx = -1i32;
                let mut second_forward_hint = 0u32;
                for i in 0..REFS_PER_FRAME {
                    let ref_hint = self.refs[fh.ref_frame_idx[i] as usize].order_hint;
                    if relative_dist(seq, ref_hint, forward_hint) < 0
                        && (second_forward_idx < 0
                            || relative_dist(seq, ref_hint, second_forward_hint) > 0)
                    {
                        second_forward_idx = i as i32;
                        second_forward_hint = ref_hint;
                    }
                }
                if second_forward_idx < 0 {
                    false
                } else {
                    fh.skip_mode_frame = [
                        (LAST_FRAME as i32 + forward_idx.min(second_forward_idx)) as u8,
                        (LAST_FRAME as i32 + forward_idx.max(second_forward_idx)) as u8,
                    ];
                    true
                }
            }
        };
        fh.skip_mode_present = if allowed { r.flag()? } else { false };
        Ok(())
    }

    fn global_motion_params(
        fh: &mut FrameHeader,
        r: &mut BitReader,
        prev_gm_params: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
    ) -> Result<()> {
        fh.global_motion = GlobalMotion::default();
        if fh.frame_is_intra() {
            return Ok(());
        }
        const IDENTITY: u8 = 0;
        const TRANSLATION: u8 = 1;
        const ROTZOOM: u8 = 2;
        const AFFINE: u8 = 3;
        let high_precision = fh.allow_high_precision_mv;
        let gm = &mut fh.global_motion;

        let read_param = |r: &mut BitReader,
                          gm_type: u8,
                          ref_frame: usize,
                          idx: usize,
                          gm_params: &mut [[i32; 6]; TOTAL_REFS_PER_FRAME]|
         -> Result<()> {
            let (mut abs_bits, mut prec_bits) = (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS);
            if idx < 2 {
                if gm_type == TRANSLATION {
                    abs_bits = GM_ABS_TRANS_ONLY_BITS - !high_precision as u32;
                    prec_bits = GM_TRANS_ONLY_PREC_BITS - !high_precision as u32;
                } else {
                    abs_bits = GM_ABS_TRANS_BITS;
                    prec_bits = GM_TRANS_PREC_BITS;
                }
            }
            let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
            let (round, sub) = if idx % 3 == 2 {
                (1i32 << WARPEDMODEL_PREC_BITS, 1i32 << prec_bits)
            } else {
                (0, 0)
            };
            let mx = 1i32 << abs_bits;
            let rr = (prev_gm_params[ref_frame][idx] >> prec_diff) - sub;
            gm_params[ref_frame][idx] =
                (r.signed_subexp_with_ref(-mx, mx + 1, rr)? << prec_diff) + round;
            Ok(())
        };

        for ref_frame in LAST_FRAME..TOTAL_REFS_PER_FRAME {
            let gm_type = if r.flag()? {
                if r.flag()? {
                    ROTZOOM
                } else if r.flag()? {
                    TRANSLATION
                } else {
                    AFFINE
                }
            } else {
                IDENTITY
            };
            gm.gm_type[ref_frame] = gm_type;
            if gm_type >= ROTZOOM {
                read_param(r, gm_type, ref_frame, 2, &mut gm.gm_params)?;
                read_param(r, gm_type, ref_frame, 3, &mut gm.gm_params)?;
                if gm_type == AFFINE {
                    read_param(r, gm_type, ref_frame, 4, &mut gm.gm_params)?;
                    read_param(r, gm_type, ref_frame, 5, &mut gm.gm_params)?;
                } else {
                    gm.gm_params[ref_frame][4] = -gm.gm_params[ref_frame][3];
                    gm.gm_params[ref_frame][5] = gm.gm_params[ref_frame][2];
                }
            }
            if gm_type >= TRANSLATION {
                read_param(r, gm_type, ref_frame, 0, &mut gm.gm_params)?;
                read_param(r, gm_type, ref_frame, 1, &mut gm.gm_params)?;
            }
        }
        Ok(())
    }

    /// Read `film_grain_params()` to get past it, keeping only whether grain is
    /// to be applied.
    fn film_grain_params(
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        r: &mut BitReader,
    ) -> Result<()> {
        if !seq.film_grain_params_present || (!fh.show_frame && !fh.showable_frame) {
            return Ok(());
        }
        fh.apply_grain = r.flag()?;
        if !fh.apply_grain {
            return Ok(());
        }
        r.skip(16)?; // grain_seed
        let update_grain = if fh.frame_type == INTER_FRAME {
            r.flag()?
        } else {
            true
        };
        if !update_grain {
            r.skip(3)?; // film_grain_params_ref_idx
            return Ok(());
        }
        let num_y_points = r.bits(4)?;
        r.skip(16 * num_y_points as usize)?;
        let chroma_scaling_from_luma = if seq.color.mono_chrome {
            false
        } else {
            r.flag()?
        };
        let (mut num_cb_points, mut num_cr_points) = (0, 0);
        if !(seq.color.mono_chrome
            || chroma_scaling_from_luma
            || (seq.color.subsampling_x && seq.color.subsampling_y && num_y_points == 0))
        {
            num_cb_points = r.bits(4)?;
            r.skip(16 * num_cb_points as usize)?;
            num_cr_points = r.bits(4)?;
            r.skip(16 * num_cr_points as usize)?;
        }
        r.skip(2)?; // grain_scaling_minus_8
        let ar_coeff_lag = r.bits(2)?;
        let num_pos_luma = 2 * ar_coeff_lag * (ar_coeff_lag + 1);
        let num_pos_chroma = if num_y_points > 0 {
            r.skip(8 * num_pos_luma as usize)?;
            num_pos_luma + 1
        } else {
            num_pos_luma
        };
        if chroma_scaling_from_luma || num_cb_points > 0 {
            r.skip(8 * num_pos_chroma as usize)?;
        }
        if chroma_scaling_from_luma || num_cr_points > 0 {
            r.skip(8 * num_pos_chroma as usize)?;
        }
        r.skip(4)?; // ar_coeff_shift_minus_6, grain_scale_shift
        if num_cb_points > 0 {
            r.skip(8 + 8 + 9)?;
        }
        if num_cr_points > 0 {
            r.skip(8 + 8 + 9)?;
        }
        r.skip(2)?; // overlap_flag, clip_to_restricted_range
        Ok(())
    }

    /// Parse a `tile_group_obu()` payload against the frame header it belongs
    /// to. Tile offsets are relative to the start of `payload`.
    pub fn parse_tile_group(&self, fh: &FrameHeader, payload: &[u8]) -> Result<TileGroup> {
        let t = &fh.tile_info;
        let num_tiles = t.tile_rows * t.tile_cols;
        let mut r = BitReader::new(payload);
        let mut tile_start_and_end_present = false;
        if num_tiles > 1 {
            tile_start_and_end_present = r.flag()?;
        }
        let (tg_start, tg_end) = if num_tiles == 1 || !tile_start_and_end_present {
            (0, num_tiles - 1)
        } else {
            let bits = t.tile_cols_log2 + t.tile_rows_log2;
            (r.bits(bits)?, r.bits(bits)?)
        };
        if tg_end < tg_start || tg_end >= num_tiles {
            return Err(err("tile group names tiles that do not exist"));
        }
        r.byte_align();
        let mut pos = r.position() / 8;
        let mut tiles = Vec::new();
        for tile_num in tg_start..=tg_end {
            let size = if tile_num == tg_end {
                payload
                    .len()
                    .checked_sub(pos)
                    .ok_or_else(|| err("tile group has no data"))?
            } else {
                let mut sr = BitReader::new(&payload[pos.min(payload.len())..]);
                let size = sr.le(t.tile_size_bytes)? as usize + 1;
                pos += t.tile_size_bytes as usize;
                size
            };
            if pos + size > payload.len() {
                return Err(err("a tile extends past its tile group"));
            }
            tiles.push(TileData { offset: pos, size });
            pos += size;
        }
        Ok(TileGroup { tg_end, tiles })
    }

    /// Whether `payload` begins a key frame that is shown, which is where
    /// decoding can start. Does no state updates.
    pub fn starts_with_shown_key_frame(&self, payload: &[u8]) -> bool {
        let Some(seq) = &self.seq else {
            return false;
        };
        let mut r = BitReader::new(payload);
        if seq.reduced_still_picture_header {
            return true;
        }
        let Ok(show_existing) = r.flag() else {
            return false;
        };
        if show_existing {
            return false;
        }
        let (Ok(frame_type), Ok(show_frame)) = (r.bits(2), r.flag()) else {
            return false;
        };
        frame_type == KEY_FRAME as u32 && show_frame
    }
}
