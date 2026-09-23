//! Packed binary hypervectors from Rust (crate::packed).
//!
//! **Tolerance: none.** A popcount, a majority vote and a bit rotation are
//! integer operations, so the reference is the same thing computed on
//! unpacked bits on the host and the comparison is `==`. A tolerance here
//! would mean a kernel was wrong.

use grilly_core::{DType, Device, Tensor};
use grilly_vsa::packed;

const DIM: usize = 2048;
const WORDS: usize = DIM / 32;

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

/// `n` vectors of random bits, as the bits and as packed words.
fn vectors(n: usize, seed: u64) -> (Vec<Vec<u8>>, Tensor) {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let bits: Vec<Vec<u8>> = (0..n).map(|_| (0..DIM).map(|_| (next() & 1) as u8).collect()).collect();
    let mut words = vec![0i32; n * WORDS];
    for (v, row) in bits.iter().enumerate() {
        for (i, &bit) in row.iter().enumerate() {
            if bit == 1 {
                words[v * WORDS + i / 32] |= 1i32 << (i % 32);
            }
        }
    }
    let packed = Tensor::from_bytes(
        &dev(),
        &words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(),
        &[n, WORDS],
        DType::I32,
    )
    .unwrap();
    (bits, packed)
}

fn as_i32(t: &Tensor) -> Vec<i32> {
    t.to_bytes().unwrap().chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn hamming_counts_every_differing_bit() {
    let (bits, x) = vectors(64, 1);
    let (_, one) = vectors(1, 1);                 // the same seed: row 0 again
    let query = one.reshape(&[WORDS]).unwrap();
    let got = as_i32(&packed::hamming(&query, &x).unwrap());
    let want: Vec<i32> = bits
        .iter()
        .map(|row| row.iter().zip(&bits[0]).filter(|(a, b)| a != b).count() as i32)
        .collect();
    assert_eq!(got, want);
    assert_eq!(got[0], 0, "a vector is at distance 0 from itself");
}

#[test]
fn bundling_is_a_majority_vote_and_an_even_tie_goes_to_zero() {
    for n in [3usize, 4, 5, 8] {
        let (bits, x) = vectors(n, 10 + n as u64);
        let got = as_i32(&packed::bundle(&x).unwrap());
        for i in 0..DIM {
            let set = bits.iter().filter(|row| row[i] == 1).count();
            let want = u32::from(2 * set > n);
            let have = (got[i / 32] as u32 >> (i % 32)) & 1;
            assert_eq!(have, want, "n = {n}, bit {i}: {set} of {n} set");
        }
    }
}

#[test]
fn permuting_rotates_the_bits_and_reverses() {
    let (bits, x) = vectors(1, 30);
    for shift in [0i64, 1, 31, 32, 33, DIM as i64 - 1, DIM as i64, -5] {
        let got = as_i32(&packed::permute(&x, DIM, shift).unwrap());
        for i in 0..DIM {
            let from = (i as i64 - shift).rem_euclid(DIM as i64) as usize;
            let have = (got[i / 32] as u32 >> (i % 32)) & 1;
            assert_eq!(have, u32::from(bits[0][from] == 1), "shift {shift}, bit {i}");
        }
    }
    let rotated = packed::permute(&x, DIM, 7).unwrap();
    assert_eq!(as_i32(&packed::permute(&rotated, DIM, -7).unwrap()), as_i32(&x));
}

#[test]
fn float_hypervectors_are_refused() {
    let dev = dev();
    let f = Tensor::from_slice(&dev, &vec![0.0f32; WORDS], &[1, WORDS]).unwrap();
    let e = packed::bundle(&f).unwrap_err();
    assert!(format!("{e}").contains("int32 words"), "{e}");
}
