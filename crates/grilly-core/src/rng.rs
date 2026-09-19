//! Deterministic random numbers: the same seed gives the same values on any
//! machine. They are not the values torch produces for that seed.
//!
//! The stream is Philox4x32-10 (Salmon et al., "Parallel random numbers: as
//! easy as 1, 2, 3", SC'11). Block `b` of a generator seeded with `s` is
//! `philox(counter = [lo32(b), hi32(b), 0, 0], key = [lo32(s), hi32(s)])`,
//! four u32 words. Each call to a fill method starts at the generator's
//! current block and advances it past every block it used, so a call never
//! shares blocks with another.
//!
//! - uniform in [0, 1): word `x` -> `(x >> 8) * 2^-24`; four values per block.
//! - normal: words (a, b) -> Box-Muller in f64 with
//!   `u = ((a >> 8) + 1) * 2^-24` in (0, 1] and `t = (b >> 8) * 2^-24`,
//!   giving `r cos(2 pi t)`, `r sin(2 pi t)` with `r = sqrt(-2 ln u)`, rounded
//!   to f32; four values per block.
//!
//! The only platform dependence is the f64 `ln`/`sqrt`/`cos`/`sin` in
//! Box-Muller; libm differences there are within an f64 ulp and vanish when
//! rounded to f32 except in rare ties.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// torch's default seed, so `initial_seed()` reads the same before seeding.
pub const DEFAULT_SEED: u64 = 67_280_421_310_721;

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = a as u64 * b as u64;
    ((p >> 32) as u32, p as u32)
}

/// One Philox4x32-10 block.
pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        if round > 0 {
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
        let (hi0, lo0) = mulhilo(M0, ctr[0]);
        let (hi1, lo1) = mulhilo(M1, ctr[2]);
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
    }
    ctr
}

/// A seeded random stream (the state behind `torch.Generator`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generator {
    seed: u64,
    block: u64,
}

impl Default for Generator {
    fn default() -> Self {
        Generator::new(DEFAULT_SEED)
    }
}

impl Generator {
    pub fn new(seed: u64) -> Generator {
        Generator { seed, block: 0 }
    }

    /// Restart the stream from `seed`.
    pub fn manual_seed(&mut self, seed: u64) {
        *self = Generator::new(seed);
    }

    pub fn initial_seed(&self) -> u64 {
        self.seed
    }

    /// The whole state: `(seed, next block)`.
    pub fn state(&self) -> (u64, u64) {
        (self.seed, self.block)
    }

    pub fn set_state(&mut self, (seed, block): (u64, u64)) {
        *self = Generator { seed, block };
    }

    fn next_block(&mut self) -> [u32; 4] {
        let b = self.block;
        self.block = self.block.wrapping_add(1);
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        philox4x32_10([b as u32, (b >> 32) as u32, 0, 0], key)
    }

    /// `n` values uniform in [0, 1).
    pub fn uniform(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            for w in self.next_block() {
                if out.len() < n {
                    out.push((w >> 8) as f32 * (1.0 / (1u32 << 24) as f32));
                }
            }
        }
        out
    }

    /// `n` values from the standard normal distribution.
    pub fn normal(&mut self, n: usize) -> Vec<f32> {
        const SCALE: f64 = 1.0 / (1u32 << 24) as f64;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let w = self.next_block();
            for pair in [(w[0], w[1]), (w[2], w[3])] {
                let u = ((pair.0 >> 8) as f64 + 1.0) * SCALE;
                let t = (pair.1 >> 8) as f64 * SCALE;
                let r = (-2.0 * u.ln()).sqrt();
                let theta = std::f64::consts::TAU * t;
                for z in [r * theta.cos(), r * theta.sin()] {
                    if out.len() < n {
                        out.push(z as f32);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer vectors from Random123's kat_vectors (philox4x32 10).
    #[test]
    fn philox_known_answers() {
        assert_eq!(
            philox4x32_10([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn replay_and_independence() {
        let mut a = Generator::new(7);
        let first = a.normal(10);
        a.manual_seed(7);
        assert_eq!(a.normal(10), first);
        assert_ne!(Generator::new(8).normal(10), first);
        // consecutive calls do not overlap
        let mut g = Generator::new(7);
        let (x, y) = (g.uniform(3), g.uniform(3));
        assert_ne!(x, y);
    }

    #[test]
    fn distributions_look_right() {
        let mut g = Generator::new(1);
        let n = 1 << 20;
        let u = g.uniform(n);
        assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)));
        let mean_u = u.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        assert!((mean_u - 0.5).abs() < 2e-3, "uniform mean {mean_u}");

        let z = g.normal(n);
        assert!(z.iter().all(|v| v.is_finite()));
        let mean = z.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = z.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        // 1M samples: standard error of the mean ~1e-3, of the variance ~1.4e-3
        assert!(mean.abs() < 5e-3, "normal mean {mean}");
        assert!((var - 1.0).abs() < 7e-3, "normal variance {var}");
    }
}
