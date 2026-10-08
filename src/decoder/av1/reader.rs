//! A bit reader for AV1 syntax.
//!
//! Unlike the H.26x readers this one does no emulation-prevention stripping:
//! AV1 has none, and a `0x03` byte is just data.

use crate::error::{PixelForgeError, Result};

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
}

fn invalid(what: &str) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("AV1 decode: {what}"))
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits read so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn bit(&mut self) -> Result<u32> {
        let byte = self
            .data
            .get(self.pos / 8)
            .ok_or_else(|| invalid("read past the end of an OBU"))?;
        let bit = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Ok(bit as u32)
    }

    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// `f(n)`: `n` bits, most significant first. `n` is at most 32.
    pub fn bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as u64;
        }
        Ok(v as u32)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() * 8 {
            return Err(invalid("skip past the end of an OBU"));
        }
        self.pos += n;
        Ok(())
    }

    /// `su(n)`: an `n`-bit two's complement value.
    pub fn su(&mut self, n: u32) -> Result<i32> {
        let value = self.bits(n)? as i32;
        let sign_mask = 1i32 << (n - 1);
        Ok(if value & sign_mask != 0 {
            value - 2 * sign_mask
        } else {
            value
        })
    }

    /// `ns(n)`: a value in `0..n` coded in as few bits as possible.
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.bits(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.bit()?;
        Ok((v << 1) - m + extra)
    }

    /// `uvlc()`.
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        while self.bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return Ok(u32::MAX);
            }
        }
        let value = self.bits(leading_zeros)?;
        Ok(value + (1u32 << leading_zeros) - 1)
    }

    /// `le(n)`: `n` bytes, least significant first. The reader must be byte
    /// aligned.
    pub fn le(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for i in 0..n {
            v |= self.bits(8)? << (8 * i);
        }
        Ok(v)
    }

    /// `leb128()`.
    pub fn leb128(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for i in 0..8 {
            let byte = self.bits(8)? as u64;
            value |= (byte & 0x7F) << (i * 7);
            if byte & 0x80 == 0 {
                break;
            }
        }
        Ok(value)
    }

    /// `byte_alignment()`: skip to the next byte boundary.
    pub fn byte_align(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }

    /// `decode_signed_subexp_with_ref(low, high, r)`.
    pub fn signed_subexp_with_ref(&mut self, low: i32, high: i32, r: i32) -> Result<i32> {
        let x = self.unsigned_subexp_with_ref(high - low, r - low)?;
        Ok(x + low)
    }

    fn unsigned_subexp_with_ref(&mut self, mx: i32, r: i32) -> Result<i32> {
        let v = self.subexp(mx)?;
        if (r << 1) <= mx {
            Ok(inverse_recenter(r, v))
        } else {
            Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
        }
    }

    fn subexp(&mut self, num_syms: i32) -> Result<i32> {
        let mut i = 0;
        let mut mk = 0;
        let k = 3;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1 << b2;
            if num_syms <= mk + 3 * a {
                let v = self.ns((num_syms - mk) as u32)?;
                return Ok(v as i32 + mk);
            }
            if self.flag()? {
                i += 1;
                mk += a;
            } else {
                let v = self.bits(b2 as u32)?;
                return Ok(v as i32 + mk);
            }
        }
    }
}

fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `FloorLog2(x)` for `x >= 1`.
pub(crate) fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fixed_width_and_signed_values() {
        let mut r = BitReader::new(&[0b1010_1100, 0b1111_0000]);
        assert_eq!(r.bits(4).unwrap(), 0b1010);
        assert_eq!(r.su(4).unwrap(), -4);
        assert_eq!(r.su(4).unwrap(), -1);
    }

    #[test]
    fn a_zero_byte_after_two_zeros_is_data() {
        // The H.26x reader would drop the 0x03.
        let mut r = BitReader::new(&[0x00, 0x00, 0x03]);
        assert_eq!(r.bits(24).unwrap(), 3);
    }

    #[test]
    fn leb128_reads_multi_byte_values() {
        let mut r = BitReader::new(&[0xE5, 0x8E, 0x26]);
        assert_eq!(r.leb128().unwrap(), 624_485);
    }

    #[allow(clippy::unusual_byte_groupings)]
    #[test]
    fn ns_uses_the_short_code_for_small_values() {
        // n = 5: w = 3, m = 3. Values 0..3 take 2 bits, 3 and 4 take 3.
        let mut r = BitReader::new(&[0b00_01_10_11, 0b0_111_0000]);
        assert_eq!(r.ns(5).unwrap(), 0);
        assert_eq!(r.ns(5).unwrap(), 1);
        assert_eq!(r.ns(5).unwrap(), 2);
        assert_eq!(r.ns(5).unwrap(), 3);
        assert_eq!(r.ns(5).unwrap(), 4);
    }

    #[allow(clippy::unusual_byte_groupings)]
    #[test]
    fn uvlc_decodes_exp_golomb() {
        // 1 -> 0, 010 -> 1, 011 -> 2.
        let mut r = BitReader::new(&[0b1_010_011_0]);
        assert_eq!(r.uvlc().unwrap(), 0);
        assert_eq!(r.uvlc().unwrap(), 1);
        assert_eq!(r.uvlc().unwrap(), 2);
    }

    #[test]
    fn reading_past_the_end_is_an_error() {
        let mut r = BitReader::new(&[0xFF]);
        r.bits(8).unwrap();
        assert!(r.bit().is_err());
    }
}
