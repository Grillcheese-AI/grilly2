//! The shortlist head against a plain CPU implementation of the same route.
//!
//! Tolerance: exact. The union is integers throughout (distances, first
//! positions, counts), so it must equal the reference row for row. The
//! token is compared with the reference's float64 argmax over that union;
//! the data are random uniform, where the best and second-best dot products
//! of a union are far apart next to fp32 rounding, so a disagreement is a
//! wrong row, not rounding. Cluster scores are random too, except where a
//! test makes two exactly equal on purpose (the same centroid twice), which
//! both sides must break by index.

use grilly_core::{DType, Device, Tensor};
use grilly_infer::shortlist::{ShortlistHead, Union};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn ints(dev: &Device, v: &[i32], shape: &[usize]) -> Tensor {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Tensor::from_bytes(dev, &bytes, shape, DType::I32).unwrap()
}

struct Problem {
    vocab: usize,
    dim: usize,
    w: Vec<f32>,
    cent: Vec<f32>,
    lists: Vec<Vec<usize>>,
    succ: Vec<i32>,
    width: usize,
}

fn problem(
    seed: u64,
    vocab: usize,
    dim: usize,
    clusters: usize,
    max_len: usize,
    width: usize,
) -> Problem {
    let mut rng = Rng(seed);
    let w: Vec<f32> = (0..vocab * dim).map(|_| rng.unit()).collect();
    let mut cent: Vec<f32> = (0..clusters * dim).map(|_| rng.unit()).collect();
    // cluster 3 is cluster 1 again: exactly equal scores, broken by index
    let copy: Vec<f32> = cent[dim..2 * dim].to_vec();
    cent[3 * dim..4 * dim].copy_from_slice(&copy);
    let lists = (0..clusters)
        .map(|_| {
            let len = 1 + rng.below(max_len);
            let mut seen = vec![false; vocab];
            let mut l = Vec::new();
            while l.len() < len {
                let t = rng.below(vocab);
                if !seen[t] {
                    seen[t] = true;
                    l.push(t);
                }
            }
            l
        })
        .collect();
    let succ = (0..vocab * width)
        .map(|_| {
            if rng.below(4) == 0 {
                -1
            } else {
                rng.below(vocab) as i32
            }
        })
        .collect();
    Problem {
        vocab,
        dim,
        w,
        cent,
        lists,
        succ,
        width,
    }
}

impl Problem {
    fn head(&self, dev: &Device, near: usize, context: usize, ranks: usize) -> ShortlistHead {
        let clusters = self.lists.len();
        let mut ids = Vec::new();
        let mut off = vec![0i32];
        for l in &self.lists {
            ids.extend(l.iter().map(|&t| t as i32));
            off.push(ids.len() as i32);
        }
        let weight = Tensor::from_slice(dev, &self.w, &[self.vocab, self.dim]).unwrap();
        let cent = Tensor::from_slice(dev, &self.cent, &[clusters, self.dim]).unwrap();
        let ids_t = ints(dev, &ids, &[ids.len()]);
        let off_t = ints(dev, &off, &[clusters + 1]);
        let succ = (self.width > 0).then(|| ints(dev, &self.succ, &[self.vocab, self.width]));
        ShortlistHead::new(
            &weight,
            &cent,
            &ids_t,
            &off_t,
            succ.as_ref(),
            near,
            context,
            ranks,
        )
        .unwrap()
    }

    fn dot(&self, x: &[f32], row: &[f32]) -> f64 {
        x.iter().zip(row).map(|(&a, &b)| a as f64 * b as f64).sum()
    }

    /// The route on the CPU: (union, short, token).
    fn reference(
        &self,
        x: &[f32],
        prev: Option<usize>,
        near: usize,
        context: usize,
        ranks: usize,
    ) -> (Union, usize) {
        let d = self.dim;
        let scores: Vec<f64> = (0..self.lists.len())
            .map(|c| self.dot(x, &self.cent[c * d..(c + 1) * d]))
            .collect();
        let mut order: Vec<usize> = (0..self.lists.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
        let mut seen = vec![false; self.vocab];
        let mut ctx = Vec::new();
        'ranks: for &c in order.iter().take(ranks) {
            for &t in &self.lists[c] {
                if !seen[t] {
                    seen[t] = true;
                    ctx.push(t);
                    if ctx.len() == context {
                        break 'ranks;
                    }
                }
            }
        }
        let short = ctx.len() < context;
        let mut inn = vec![false; self.vocab];
        for &t in &ctx {
            inn[t] = true;
        }
        if let Some(p) = prev {
            for s in 0..self.width {
                let t = self.succ[p * self.width + s];
                if t >= 0 {
                    inn[t as usize] = true;
                }
            }
        }
        let bits = |v: &[f32]| -> Vec<bool> { v.iter().map(|&a| a > 0.0).collect() };
        let xb = bits(x);
        let mut dist: Vec<(usize, usize)> = (0..self.vocab)
            .map(|r| {
                (
                    bits(&self.w[r * d..(r + 1) * d])
                        .iter()
                        .zip(&xb)
                        .filter(|(a, b)| a != b)
                        .count(),
                    r,
                )
            })
            .collect();
        dist.sort();
        for &(_, r) in dist.iter().take(near) {
            inn[r] = true;
        }
        let rows: Vec<u32> = (0..self.vocab as u32)
            .filter(|&r| inn[r as usize])
            .collect();
        let mut best = (f64::NEG_INFINITY, usize::MAX);
        for &r in &rows {
            let v = self.dot(x, &self.w[r as usize * d..(r as usize + 1) * d]);
            if v > best.0 {
                best = (v, r as usize);
            }
        }
        (Union { rows, short }, best.1)
    }
}

fn check(
    p: &Problem,
    near: usize,
    context: usize,
    ranks: usize,
    states: usize,
    seed: u64,
    expect_short: Option<bool>,
) {
    let dev = dev();
    let head = p.head(&dev, near, context, ranks);
    let mut rng = Rng(seed);
    for i in 0..states {
        let x: Vec<f32> = (0..p.dim).map(|_| rng.unit()).collect();
        let xt = Tensor::from_slice(&dev, &x, &[1, p.dim]).unwrap();
        let prev = if i % 2 == 0 {
            Some(rng.below(p.vocab))
        } else {
            None
        };
        let (want, token) = p.reference(&x, prev, near, context, ranks);
        let got = head.union(&xt, prev).unwrap();
        assert_eq!(got.short, want.short, "state {i}: short flag");
        if let Some(s) = expect_short {
            assert_eq!(
                got.short, s,
                "state {i}: this case is built to be short={s}"
            );
        }
        assert_eq!(got.rows.len(), want.rows.len(), "state {i}: union size");
        assert_eq!(got.rows, want.rows, "state {i}: union rows");
        let choice = head.argmax(&xt, prev).unwrap();
        assert_eq!(choice.rows, want.rows.len(), "state {i}: rows scored");
        assert_eq!(choice.token, token, "state {i}: token");
        assert_eq!(choice.short, want.short);
    }
}

#[test]
fn the_codes_are_the_rows_sign_bits() {
    let dev = dev();
    let p = problem(7, 300, 256, 8, 50, 0);
    let head = p.head(&dev, 16, 32, 8);
    let words: Vec<u32> = head
        .codes()
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for r in 0..p.vocab {
        for w in 0..p.dim / 32 {
            let mut bits = 0u32;
            for b in 0..32 {
                if p.w[r * p.dim + w * 32 + b] > 0.0 {
                    bits |= 1 << b;
                }
            }
            assert_eq!(words[r * p.dim / 32 + w], bits, "row {r} word {w}");
        }
    }
}

#[test]
fn the_union_and_token_match_the_reference() {
    // the context list is cut inside a cluster, the successors join on
    // even states, and clusters 1 and 3 tie exactly
    let p = problem(11, 5000, 256, 40, 400, 4);
    check(&p, 300, 900, 40, 24, 5, Some(false));
}

#[test]
fn ties_at_the_kth_distance_go_to_the_lower_rows() {
    // 128 bits over 6,000 rows: hundreds of rows share each distance
    let p = problem(13, 6000, 128, 16, 300, 0);
    check(&p, 777, 500, 16, 16, 9, Some(false));
}

#[test]
fn a_rank_bound_short_of_k_takes_what_it_scanned_and_says_so() {
    let p = problem(17, 4000, 256, 30, 60, 2);
    check(&p, 100, 3000, 5, 8, 3, Some(true));
}

#[test]
fn many_clusters_and_a_wide_sort() {
    // 3,000 clusters: the bitonic sort pads to 4,096
    let p = problem(19, 8000, 128, 3000, 40, 0);
    check(&p, 256, 2000, 300, 6, 21, Some(false));
}
