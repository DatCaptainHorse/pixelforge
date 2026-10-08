//! Translating parsed H.265 parameter sets into the `StdVideo*` structs Vulkan
//! takes, and the decode profile they imply.

use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;

use crate::decoder::common::DecoderCommon;
use crate::decoder::h265::parser::{DecPicBuf, Pps, ProfileTierLevel, ScalingLists, Sps, Vps};
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};

type Std = ash::vk::native::StdVideoH265ScalingLists;

/// The three parameter sets a picture is decoded under.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParameterSets {
    pub vps: Arc<Vps>,
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
}

fn unsupported(what: String) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("H.265 decode: {what}"))
}

/// The decode profile a stream's SPS asks for.
pub(crate) struct DecodeProfile {
    h265: vk::VideoDecodeH265ProfileInfoKHR<'static>,
    chroma: vk::VideoChromaSubsamplingFlagsKHR,
    depth: vk::VideoComponentBitDepthFlagsKHR,
    pub pixel_format: PixelFormat,
    pub bit_depth: BitDepth,
}

impl DecodeProfile {
    /// The profile as Vulkan takes it. Borrows `self`, which owns the codec
    /// part the struct chains to.
    pub fn info(&mut self) -> vk::VideoProfileInfoKHR<'_> {
        vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
            .chroma_subsampling(self.chroma)
            .luma_bit_depth(self.depth)
            .chroma_bit_depth(self.depth)
            .push(&mut self.h265)
    }
}

/// Whether the profile the stream names can be swapped for plain Main or
/// Main 10 without changing what decodes. True for a range extension profile
/// that carries ordinary 4:2:0 video and uses none of the extension tools,
/// which is what an encoder's "main intra" mode produces.
pub(crate) fn can_relax_profile(sps: &Sps) -> bool {
    sps.ptl.profile_idc == 4
        && sps.chroma_format_idc == 1
        && sps.range_extension == Default::default()
}

/// The decode profile for `sps`. `relaxed` asks for Main or Main 10 where
/// [`can_relax_profile`] allows it, for a device that lists only those.
pub(crate) fn decode_profile(sps: &Sps, relaxed: bool) -> Result<DecodeProfile> {
    let (pixel_format, chroma) = match sps.chroma_format_idc {
        1 => (
            PixelFormat::Yuv420,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        ),
        2 => (
            PixelFormat::Yuv422,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
        ),
        3 => (
            PixelFormat::Yuv444,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        ),
        other => {
            return Err(unsupported(format!(
                "unsupported chroma_format_idc {other}"
            )));
        }
    };
    if sps.separate_colour_plane_flag {
        return Err(unsupported(
            "separate colour planes are not supported".to_string(),
        ));
    }
    if sps.bit_depth_luma_minus8 != sps.bit_depth_chroma_minus8 {
        return Err(unsupported("luma and chroma bit depths differ".to_string()));
    }
    let (bit_depth, depth) = match sps.bit_depth_luma_minus8 {
        0 => (BitDepth::Eight, vk::VideoComponentBitDepthFlagsKHR::TYPE_8),
        2 => (BitDepth::Ten, vk::VideoComponentBitDepthFlagsKHR::TYPE_10),
        other => {
            return Err(unsupported(format!("unsupported bit depth {}", other + 8)));
        }
    };

    use ash::vk::native::*;
    let std_profile = match sps.ptl.profile_idc {
        2 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
        // Main Still Picture is Main with one picture; some drivers only list
        // Main, and it decodes the same.
        1 | 3 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        4 if relaxed && can_relax_profile(sps) && bit_depth == BitDepth::Ten => {
            StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
        }
        4 if relaxed && can_relax_profile(sps) => {
            StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
        }
        4 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
        // Streams that do not name a profile get the one their format implies.
        _ if pixel_format == PixelFormat::Yuv420 && bit_depth == BitDepth::Eight => {
            StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
        }
        _ if pixel_format == PixelFormat::Yuv420 => {
            StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
        }
        _ => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
    };
    Ok(DecodeProfile {
        h265: vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(std_profile),
        chroma,
        depth,
        pixel_format,
        bit_depth,
    })
}

fn std_level(level_idc: u8) -> ash::vk::native::StdVideoH265LevelIdc {
    use ash::vk::native::*;
    match level_idc {
        0..=30 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
        31..=60 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
        61..=63 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
        64..=90 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
        91..=93 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
        94..=120 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
        121..=123 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
        124..=150 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
        151..=153 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        154..=156 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
        157..=180 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
        181..=183 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
        _ => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
    }
}

fn std_profile_tier_level(ptl: &ProfileTierLevel) -> ash::vk::native::StdVideoH265ProfileTierLevel {
    use ash::vk::native::*;
    let mut flags: StdVideoH265ProfileTierLevelFlags = unsafe { std::mem::zeroed() };
    flags.set_general_tier_flag(ptl.tier_flag as u32);
    flags.set_general_progressive_source_flag(ptl.progressive_source_flag as u32);
    flags.set_general_interlaced_source_flag(ptl.interlaced_source_flag as u32);
    flags.set_general_non_packed_constraint_flag(ptl.non_packed_constraint_flag as u32);
    flags.set_general_frame_only_constraint_flag(ptl.frame_only_constraint_flag as u32);
    StdVideoH265ProfileTierLevel {
        flags,
        general_profile_idc: match ptl.profile_idc {
            1 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
            2 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
            3 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE,
            4 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
            _ => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_INVALID,
        },
        general_level_idc: std_level(ptl.level_idc),
    }
}

fn std_dec_pic_buf(buf: &DecPicBuf) -> ash::vk::native::StdVideoH265DecPicBufMgr {
    ash::vk::native::StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: buf.max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: buf.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: buf.max_num_reorder_pics,
    }
}

fn std_scaling_lists(lists: &ScalingLists) -> Std {
    Std {
        ScalingList4x4: lists.list_4x4,
        ScalingList8x8: lists.list_8x8,
        ScalingList16x16: lists.list_16x16,
        ScalingList32x32: lists.list_32x32,
        ScalingListDCCoef16x16: lists.dc_16x16,
        ScalingListDCCoef32x32: lists.dc_32x32,
    }
}

fn std_short_term_sets(sps: &Sps) -> Vec<ash::vk::native::StdVideoH265ShortTermRefPicSet> {
    use ash::vk::native::*;
    sps.short_term_rps
        .iter()
        .map(|rps| {
            let mut flags: StdVideoH265ShortTermRefPicSetFlags = unsafe { std::mem::zeroed() };
            flags.set_inter_ref_pic_set_prediction_flag(
                rps.inter_ref_pic_set_prediction_flag as u32,
            );
            flags.set_delta_rps_sign(rps.delta_rps_sign as u32);

            let mask = |used: &[bool; 16]| {
                used.iter()
                    .enumerate()
                    .fold(0u16, |m, (i, &u)| m | ((u as u16) << i))
            };
            // The deltas as the syntax of a directly coded set gives them: the
            // distance to the previous entry, less one.
            let mut s0 = [0u16; 16];
            let mut s1 = [0u16; 16];
            let mut prev = 0;
            for (out, &delta) in s0
                .iter_mut()
                .zip(&rps.delta_poc_s0[..rps.num_negative_pics as usize])
            {
                *out = (prev - delta - 1) as u16;
                prev = delta;
            }
            let mut prev = 0;
            for (out, &delta) in s1
                .iter_mut()
                .zip(&rps.delta_poc_s1[..rps.num_positive_pics as usize])
            {
                *out = (delta - prev - 1) as u16;
                prev = delta;
            }
            StdVideoH265ShortTermRefPicSet {
                flags,
                delta_idx_minus1: rps.delta_idx_minus1,
                use_delta_flag: rps.use_delta_flag,
                abs_delta_rps_minus1: rps.abs_delta_rps_minus1 as u16,
                used_by_curr_pic_flag: rps.used_by_curr_pic_flag,
                used_by_curr_pic_s0_flag: mask(&rps.used_by_curr_pic_s0),
                used_by_curr_pic_s1_flag: mask(&rps.used_by_curr_pic_s1),
                reserved1: 0,
                reserved2: 0,
                reserved3: 0,
                num_negative_pics: rps.num_negative_pics,
                num_positive_pics: rps.num_positive_pics,
                delta_poc_s0_minus1: s0,
                delta_poc_s1_minus1: s1,
            }
        })
        .collect()
}

/// Tile column widths and row heights, minus one, for every tile but the last
/// in each direction. Uniform spacing is resolved here, so the driver gets the
/// sizes whichever way the stream expressed them.
fn tile_sizes(pps: &Pps, sps: &Sps) -> ([u16; 19], [u16; 21]) {
    let mut columns = [0u16; 19];
    let mut rows = [0u16; 21];
    if !pps.tiles_enabled_flag {
        return (columns, rows);
    }
    let cols = pps.num_tile_columns_minus1 as u32 + 1;
    let rws = pps.num_tile_rows_minus1 as u32 + 1;
    if pps.uniform_spacing_flag {
        let (w, h) = (sps.pic_width_in_ctbs(), sps.pic_height_in_ctbs());
        for i in 0..cols - 1 {
            columns[i as usize] = (((i + 1) * w) / cols - (i * w) / cols - 1) as u16;
        }
        for i in 0..rws - 1 {
            rows[i as usize] = (((i + 1) * h) / rws - (i * h) / rws - 1) as u16;
        }
    } else {
        for (dst, &src) in columns.iter_mut().zip(&pps.column_width_minus1) {
            *dst = src;
        }
        for (dst, &src) in rows.iter_mut().zip(&pps.row_height_minus1) {
            *dst = src;
        }
    }
    (columns, rows)
}

/// Create the Vulkan session parameters holding `sets`.
pub(crate) fn build(
    common: &DecoderCommon,
    session: vk::VideoSessionKHR,
    sets: &ParameterSets,
) -> Result<vk::VideoSessionParametersKHR> {
    use ash::vk::native::*;
    let (vps, sps, pps) = (&sets.vps, &sets.sps, &sets.pps);

    // --- VPS ---
    let vps_ptl = std_profile_tier_level(&vps.ptl);
    let vps_buf = std_dec_pic_buf(&vps.dec_pic_buf);
    let mut vps_flags: StdVideoH265VpsFlags = unsafe { std::mem::zeroed() };
    vps_flags.set_vps_temporal_id_nesting_flag(vps.temporal_id_nesting_flag as u32);
    vps_flags.set_vps_sub_layer_ordering_info_present_flag(
        vps.sub_layer_ordering_info_present_flag as u32,
    );
    let std_vps = StdVideoH265VideoParameterSet {
        flags: vps_flags,
        vps_video_parameter_set_id: vps.vps_id,
        vps_max_sub_layers_minus1: vps.max_sub_layers_minus1,
        reserved1: 0,
        reserved2: 0,
        vps_num_units_in_tick: 0,
        vps_time_scale: 0,
        vps_num_ticks_poc_diff_one_minus1: 0,
        reserved3: 0,
        pDecPicBufMgr: &vps_buf,
        pHrdParameters: std::ptr::null(),
        pProfileTierLevel: &vps_ptl,
    };

    // --- SPS ---
    let sps_ptl = std_profile_tier_level(&sps.ptl);
    let sps_buf = std_dec_pic_buf(&sps.dec_pic_buf);
    let sps_lists = sps.scaling_lists.as_ref().map(std_scaling_lists);
    let short_term_sets = std_short_term_sets(sps);
    let mut lt_poc_lsb = [0u32; 32];
    let mut lt_used_mask = 0u32;
    for (i, (&lsb, &used)) in sps
        .lt_ref_pic_poc_lsb_sps
        .iter()
        .zip(&sps.used_by_curr_pic_lt_sps_flag)
        .enumerate()
    {
        lt_poc_lsb[i] = lsb;
        lt_used_mask |= (used as u32) << i;
    }
    let long_term = StdVideoH265LongTermRefPicsSps {
        used_by_curr_pic_lt_sps_flag: lt_used_mask,
        lt_ref_pic_poc_lsb_sps: lt_poc_lsb,
    };

    let mut sps_flags: StdVideoH265SpsFlags = unsafe { std::mem::zeroed() };
    sps_flags.set_sps_temporal_id_nesting_flag(sps.temporal_id_nesting_flag as u32);
    sps_flags.set_separate_colour_plane_flag(sps.separate_colour_plane_flag as u32);
    sps_flags.set_conformance_window_flag(sps.conformance_window_flag as u32);
    sps_flags.set_sps_sub_layer_ordering_info_present_flag(
        sps.sub_layer_ordering_info_present_flag as u32,
    );
    sps_flags.set_scaling_list_enabled_flag(sps.scaling_list_enabled_flag as u32);
    sps_flags.set_sps_scaling_list_data_present_flag(sps.sps_scaling_list_data_present_flag as u32);
    sps_flags.set_amp_enabled_flag(sps.amp_enabled_flag as u32);
    sps_flags
        .set_sample_adaptive_offset_enabled_flag(sps.sample_adaptive_offset_enabled_flag as u32);
    sps_flags.set_pcm_enabled_flag(sps.pcm_enabled_flag as u32);
    sps_flags.set_pcm_loop_filter_disabled_flag(sps.pcm_loop_filter_disabled_flag as u32);
    sps_flags.set_long_term_ref_pics_present_flag(sps.long_term_ref_pics_present_flag as u32);
    sps_flags.set_sps_temporal_mvp_enabled_flag(sps.sps_temporal_mvp_enabled_flag as u32);
    sps_flags
        .set_strong_intra_smoothing_enabled_flag(sps.strong_intra_smoothing_enabled_flag as u32);
    let ext = &sps.range_extension;
    if ext.present {
        sps_flags.set_sps_extension_present_flag(1);
        sps_flags.set_sps_range_extension_flag(1);
        sps_flags.set_transform_skip_rotation_enabled_flag(
            ext.transform_skip_rotation_enabled_flag as u32,
        );
        sps_flags.set_transform_skip_context_enabled_flag(
            ext.transform_skip_context_enabled_flag as u32,
        );
        sps_flags.set_implicit_rdpcm_enabled_flag(ext.implicit_rdpcm_enabled_flag as u32);
        sps_flags.set_explicit_rdpcm_enabled_flag(ext.explicit_rdpcm_enabled_flag as u32);
        sps_flags
            .set_extended_precision_processing_flag(ext.extended_precision_processing_flag as u32);
        sps_flags.set_intra_smoothing_disabled_flag(ext.intra_smoothing_disabled_flag as u32);
        sps_flags.set_high_precision_offsets_enabled_flag(
            ext.high_precision_offsets_enabled_flag as u32,
        );
        sps_flags.set_persistent_rice_adaptation_enabled_flag(
            ext.persistent_rice_adaptation_enabled_flag as u32,
        );
        sps_flags.set_cabac_bypass_alignment_enabled_flag(
            ext.cabac_bypass_alignment_enabled_flag as u32,
        );
    }

    let std_sps = StdVideoH265SequenceParameterSet {
        flags: sps_flags,
        chroma_format_idc: sps.chroma_format_idc as StdVideoH265ChromaFormatIdc,
        pic_width_in_luma_samples: sps.pic_width_in_luma_samples,
        pic_height_in_luma_samples: sps.pic_height_in_luma_samples,
        sps_video_parameter_set_id: sps.vps_id,
        sps_max_sub_layers_minus1: sps.max_sub_layers_minus1,
        sps_seq_parameter_set_id: sps.sps_id,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        log2_min_luma_coding_block_size_minus3: sps.log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2: sps.log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size: sps
            .log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
        num_short_term_ref_pic_sets: short_term_sets.len() as u8,
        num_long_term_ref_pics_sps: sps.lt_ref_pic_poc_lsb_sps.len() as u8,
        pcm_sample_bit_depth_luma_minus1: sps.pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1: sps.pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3: sps.log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size: sps
            .log2_diff_max_min_pcm_luma_coding_block_size,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: 0,
        delta_palette_max_predictor_size: 0,
        motion_vector_resolution_control_idc: 0,
        sps_num_palette_predictor_initializers_minus1: 0,
        conf_win_left_offset: sps.conf_win_left_offset,
        conf_win_right_offset: sps.conf_win_right_offset,
        conf_win_top_offset: sps.conf_win_top_offset,
        conf_win_bottom_offset: sps.conf_win_bottom_offset,
        pProfileTierLevel: &sps_ptl,
        pDecPicBufMgr: &sps_buf,
        pScalingLists: sps_lists
            .as_ref()
            .map_or(std::ptr::null(), |l| l as *const _),
        pShortTermRefPicSet: if short_term_sets.is_empty() {
            std::ptr::null()
        } else {
            short_term_sets.as_ptr()
        },
        pLongTermRefPicsSps: if sps.long_term_ref_pics_present_flag {
            &long_term
        } else {
            std::ptr::null()
        },
        pSequenceParameterSetVui: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    };

    // --- PPS ---
    let pps_lists = pps.scaling_lists.as_ref().map(std_scaling_lists);
    let (column_width_minus1, row_height_minus1) = tile_sizes(pps, sps);
    let mut pps_flags: StdVideoH265PpsFlags = unsafe { std::mem::zeroed() };
    pps_flags.set_dependent_slice_segments_enabled_flag(
        pps.dependent_slice_segments_enabled_flag as u32,
    );
    pps_flags.set_output_flag_present_flag(pps.output_flag_present_flag as u32);
    pps_flags.set_sign_data_hiding_enabled_flag(pps.sign_data_hiding_enabled_flag as u32);
    pps_flags.set_cabac_init_present_flag(pps.cabac_init_present_flag as u32);
    pps_flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag as u32);
    pps_flags.set_transform_skip_enabled_flag(pps.transform_skip_enabled_flag as u32);
    pps_flags.set_cu_qp_delta_enabled_flag(pps.cu_qp_delta_enabled_flag as u32);
    pps_flags.set_pps_slice_chroma_qp_offsets_present_flag(
        pps.pps_slice_chroma_qp_offsets_present_flag as u32,
    );
    pps_flags.set_weighted_pred_flag(pps.weighted_pred_flag as u32);
    pps_flags.set_weighted_bipred_flag(pps.weighted_bipred_flag as u32);
    pps_flags.set_transquant_bypass_enabled_flag(pps.transquant_bypass_enabled_flag as u32);
    pps_flags.set_tiles_enabled_flag(pps.tiles_enabled_flag as u32);
    pps_flags.set_entropy_coding_sync_enabled_flag(pps.entropy_coding_sync_enabled_flag as u32);
    pps_flags.set_uniform_spacing_flag(pps.uniform_spacing_flag as u32);
    pps_flags.set_loop_filter_across_tiles_enabled_flag(
        pps.loop_filter_across_tiles_enabled_flag as u32,
    );
    pps_flags.set_pps_loop_filter_across_slices_enabled_flag(
        pps.pps_loop_filter_across_slices_enabled_flag as u32,
    );
    pps_flags.set_deblocking_filter_control_present_flag(
        pps.deblocking_filter_control_present_flag as u32,
    );
    pps_flags.set_deblocking_filter_override_enabled_flag(
        pps.deblocking_filter_override_enabled_flag as u32,
    );
    pps_flags
        .set_pps_deblocking_filter_disabled_flag(pps.pps_deblocking_filter_disabled_flag as u32);
    pps_flags.set_pps_scaling_list_data_present_flag(pps.pps_scaling_list_data_present_flag as u32);
    pps_flags.set_lists_modification_present_flag(pps.lists_modification_present_flag as u32);
    pps_flags.set_slice_segment_header_extension_present_flag(
        pps.slice_segment_header_extension_present_flag as u32,
    );
    let pext = &pps.range_extension;
    if pext.present {
        pps_flags.set_pps_extension_present_flag(1);
        pps_flags.set_pps_range_extension_flag(1);
        pps_flags.set_cross_component_prediction_enabled_flag(
            pext.cross_component_prediction_enabled_flag as u32,
        );
        pps_flags
            .set_chroma_qp_offset_list_enabled_flag(pext.chroma_qp_offset_list_enabled_flag as u32);
    }
    let std_pps = StdVideoH265PictureParameterSet {
        flags: pps_flags,
        pps_pic_parameter_set_id: pps.pps_id,
        pps_seq_parameter_set_id: pps.sps_id,
        sps_video_parameter_set_id: sps.vps_id,
        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        init_qp_minus26: pps.init_qp_minus26,
        diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth,
        pps_cb_qp_offset: pps.pps_cb_qp_offset,
        pps_cr_qp_offset: pps.pps_cr_qp_offset,
        pps_beta_offset_div2: pps.pps_beta_offset_div2,
        pps_tc_offset_div2: pps.pps_tc_offset_div2,
        log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level_minus2,
        log2_max_transform_skip_block_size_minus2: pext.log2_max_transform_skip_block_size_minus2,
        diff_cu_chroma_qp_offset_depth: pext.diff_cu_chroma_qp_offset_depth,
        chroma_qp_offset_list_len_minus1: pext.chroma_qp_offset_list_len_minus1,
        cb_qp_offset_list: pext.cb_qp_offset_list,
        cr_qp_offset_list: pext.cr_qp_offset_list,
        log2_sao_offset_scale_luma: pext.log2_sao_offset_scale_luma,
        log2_sao_offset_scale_chroma: pext.log2_sao_offset_scale_chroma,
        pps_act_y_qp_offset_plus5: 0,
        pps_act_cb_qp_offset_plus5: 0,
        pps_act_cr_qp_offset_plus3: 0,
        pps_num_palette_predictor_initializers: 0,
        luma_bit_depth_entry_minus8: 0,
        chroma_bit_depth_entry_minus8: 0,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1,
        row_height_minus1,
        reserved3: 0,
        pScalingLists: pps_lists
            .as_ref()
            .map_or(std::ptr::null(), |l| l as *const _),
        pPredictorPaletteEntries: std::ptr::null(),
    };

    let vps_array = [std_vps];
    let sps_array = [std_sps];
    let pps_array = [std_pps];
    let add_info = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
        .std_vp_ss(&vps_array)
        .std_sp_ss(&sps_array)
        .std_pp_ss(&pps_array);
    let mut h265_create = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
        .max_std_vps_count(1)
        .max_std_sps_count(1)
        .max_std_pps_count(1)
        .parameters_add_info(&add_info);
    let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
        .video_session(session)
        .push(&mut h265_create);

    unsafe {
        common
            .video_queue_fn
            .create_video_session_parameters(&create_info, None)
    }
    .map_err(|e| {
        PixelForgeError::SessionParametersCreation(format!(
            "H.265 decode session parameters: {e:?}"
        ))
    })
}
