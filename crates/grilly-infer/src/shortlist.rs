//! A vocabulary shortlist head: greedy decoding's next token from a large
//! output head without scoring every row of it.
//!
//! A model with a 128,000-token vocabulary scores all 128,000 rows of its
//! output head for every token it writes. This head scores only the union
//! of two short lists, and picks the best of those exactly (fp32 dot
//! products against the trained rows, nothing approximated in the scores):
//!
//! - **a context list**: the head's rows are grouped into clusters, each
//!   with a centroid and a list of tokens. The clusters are ranked by
//!   `x . centroid` (score descending, index ascending on a tie), and the
//!   list is their tokens in that order, without repeats, until `context`
//!   distinct tokens; plus the previous token's successors, if given;
//! - **the `near` rows nearest the state by Hamming distance** between sign
//!   bits — the state's (`x > 0`) and each row's, packed when the head is
//!   made — ties to the lower row.
//!
//! The whole step runs on the GPU in eight dispatches, from the state to
//! the token, and reads back 16 bytes: no host list, no sort, no gather.
//!
//! 1. `g2-shortlist-context` — the centroid scores; packs the state's bits
//!    and clears the histogram;
//! 2. `g2-shortlist-hamming` — every row's distance, and their histogram;
//! 3. `g2-shortlist-rank` — the clusters ranked, each finding its own rank
//!    by counting; the `near`-th distance, from the histogram;
//! 4. `g2-shortlist-mark` — each listed token's first position; the rows at
//!    the `near`-th distance counted per block;
//! 5. `g2-shortlist-count` — each cluster's new tokens;
//! 6. `g2-shortlist-cut` — where the context list stops; which tied rows
//!    are taken (the lowest); the successors;
//! 7. `g2-shortlist-score` — the union's rows scored, a best per workgroup;
//! 8. `g2-shortlist-best` — the best overall (`torch.argmax`'s first index
//!    on a tie).
//!
//! Clusters are scanned to rank `ranks` at most. A state whose first
//! `ranks` clusters hold fewer than `context` distinct tokens takes all of
//! them and is reported as short ([`Choice::short`]); nothing is hidden.
//!
//! The ranking is on the GPU's fp32 scores, so two clusters whose scores
//! are within rounding of each other can rank in the other order than a
//! different implementation's (numpy's, say) and change the list. The union
//! is otherwise exact: distances are integers, and the cut is a count.

use grilly_core::{Arg, DType, Error, Kernel, Result, Tensor};

const PACK: Kernel = Kernel {
    shader: "g2-shortlist-pack",
    bindings: 2,
    push_words: 4,
    local_size: [256, 1, 1],
};
const CONTEXT: Kernel = Kernel {
    shader: "g2-shortlist-context",
    bindings: 5,
    push_words: 3,
    local_size: [256, 1, 1],
};
const HAMMING: Kernel = Kernel {
    shader: "g2-shortlist-hamming",
    bindings: 5,
    push_words: 3,
    local_size: [256, 1, 1],
};
const RANK: Kernel = Kernel {
    shader: "g2-shortlist-rank",
    bindings: 3,
    push_words: 5,
    local_size: [256, 1, 1],
};
const MARK: Kernel = Kernel {
    shader: "g2-shortlist-mark",
    bindings: 6,
    push_words: 3,
    local_size: [256, 1, 1],
};
const COUNT: Kernel = Kernel {
    shader: "g2-shortlist-count",
    bindings: 5,
    push_words: 1,
    local_size: [256, 1, 1],
};
const CUT: Kernel = Kernel {
    shader: "g2-shortlist-cut",
    bindings: 8,
    push_words: 8,
    local_size: [1024, 1, 1],
};
const SCORE: Kernel = Kernel {
    shader: "g2-shortlist-score",
    bindings: 6,
    push_words: 4,
    local_size: [256, 1, 1],
};
const BEST: Kernel = Kernel {
    shader: "g2-shortlist-best",
    bindings: 3,
    push_words: 1,
    local_size: [1024, 1, 1],
};
pub const KERNELS: &[Kernel] = &[PACK, CONTEXT, HAMMING, RANK, MARK, COUNT, CUT, SCORE, BEST];

/// The step's dispatches, in order (what [`ShortlistHead::profile`] counts).
pub const STAGES: usize = 8;

/// Limits the kernels are written for.
pub const MAX_DIM: usize = 4096;
pub const MAX_CLUSTERS: usize = 4096;
pub const MAX_RANKS: usize = 1024;

const PLAN_ORDER: usize = 16;
const CONTEXT_ROWS: usize = 8; // centroids per workgroup, g2-shortlist-context
const HAMMING_TILE: usize = 512; // rows per workgroup, g2-shortlist-hamming
const SCORE_ROWS: usize = 64; // rows per workgroup, g2-shortlist-score
const RANK_GROUP: usize = 256; // clusters per workgroup, g2-shortlist-rank
const TIE_BLOCK: usize = 512; // rows per tie count, at least (g2-shortlist-mark)
const NONE: u32 = u32::MAX;

/// One step's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choice {
    /// The best row of the union: the next token.
    pub token: usize,
    /// How many rows were scored (the union's size).
    pub rows: usize,
    /// The ranks scanned held fewer than `context` distinct tokens.
    pub short: bool,
}

/// The union a step would score, read back for checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Union {
    /// Its rows, ascending.
    pub rows: Vec<u32>,
    pub short: bool,
}

pub struct ShortlistHead {
    weight: Tensor,
    codes: Tensor,
    centroids: Tensor,
    cluster_ids: Tensor,
    cluster_offsets: Tensor,
    successors: Option<Tensor>,
    vocab: usize,
    dim: usize,
    clusters: usize,
    succ_width: usize,
    near: usize,
    context: usize,
    ranks: usize,
    // scratch, reused every step
    q: Tensor,
    scores: Tensor,
    dist: Tensor,
    firstpos: Tensor,
    hist: Tensor,
    plan: Tensor,
    cnt: Tensor,
    tiecnt: Tensor,
    part: Tensor,
    out: Tensor,
    tie_block: usize,
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

fn words_of(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_bytes()?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// `t` with its data at offset 0 (the kernels index these from the start).
fn at_zero(t: &Tensor) -> Result<Tensor> {
    if t.offset() == 0 {
        Ok(t.clone())
    } else {
        t.clone_data()
    }
}

impl ShortlistHead {
    /// A head over `weight` (vocab, dim), float32, kept by reference (not
    /// copied). `centroids` (clusters, dim) float32; `cluster_ids` int32,
    /// every cluster's tokens one list after another, no repeats within a
    /// list; `cluster_offsets` int32 (clusters + 1,), cluster c's list is
    /// `cluster_ids[offsets[c]..offsets[c + 1]]`; `successors` int32
    /// (vocab, width), -1 for none.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: &Tensor,
        centroids: &Tensor,
        cluster_ids: &Tensor,
        cluster_offsets: &Tensor,
        successors: Option<&Tensor>,
        near: usize,
        context: usize,
        ranks: usize,
    ) -> Result<ShortlistHead> {
        let bad = |m: String| Err(Error::Shape(m));
        let (vocab, dim) = match (weight.dtype(), weight.shape()) {
            (DType::F32, &[v, d]) if v > 0 => (v, d),
            _ => {
                return bad(format!(
                    "shortlist head: weight must be float32 (vocab, dim), got {} {:?}",
                    weight.dtype().name(),
                    weight.shape()
                ))
            }
        };
        if dim == 0 || dim % 128 != 0 || dim > MAX_DIM {
            return bad(format!(
                "shortlist head: dim must be a multiple of 128 up to {MAX_DIM}, got {dim}"
            ));
        }
        let clusters = match (centroids.dtype(), centroids.shape()) {
            (DType::F32, &[c, d]) if d == dim && (1..=MAX_CLUSTERS).contains(&c) => c,
            _ => return bad(format!("shortlist head: centroids must be float32 (1..={MAX_CLUSTERS}, {dim}), got {} {:?}", centroids.dtype().name(), centroids.shape())),
        };
        if cluster_offsets.dtype() != DType::I32 || cluster_offsets.shape() != [clusters + 1] {
            return bad(format!(
                "shortlist head: cluster_offsets must be int32 ({},), got {} {:?}",
                clusters + 1,
                cluster_offsets.dtype().name(),
                cluster_offsets.shape()
            ));
        }
        if cluster_ids.dtype() != DType::I32 || cluster_ids.dim() != 1 {
            return bad(format!(
                "shortlist head: cluster_ids must be int32 (n,), got {} {:?}",
                cluster_ids.dtype().name(),
                cluster_ids.shape()
            ));
        }
        let offsets = words_of(cluster_offsets)?;
        if offsets[0] != 0
            || offsets.windows(2).any(|w| w[1] < w[0])
            || offsets[clusters] as usize != cluster_ids.numel()
        {
            return Err(Error::Invalid(
                "shortlist head: cluster_offsets must rise from 0 to len(cluster_ids)".into(),
            ));
        }
        if words_of(cluster_ids)?.iter().any(|&i| i as usize >= vocab) {
            return Err(Error::Invalid(format!(
                "shortlist head: cluster_ids must be tokens below {vocab}"
            )));
        }
        let succ_width = match successors {
            None => 0,
            Some(s) => match (s.dtype(), s.shape()) {
                (DType::I32, &[v, w]) if v == vocab => w,
                _ => {
                    return bad(format!(
                        "shortlist head: successors must be int32 ({vocab}, width), got {} {:?}",
                        s.dtype().name(),
                        s.shape()
                    ))
                }
            },
        };
        if succ_width > 1024 {
            return bad(format!(
                "shortlist head: at most 1024 successors a token, got {succ_width}"
            ));
        }
        if near == 0 || context == 0 {
            return Err(Error::Invalid(
                "shortlist head: near and context must be positive".into(),
            ));
        }
        let ranks = ranks.min(clusters);
        if ranks == 0 || ranks > MAX_RANKS {
            return Err(Error::Invalid(format!(
                "shortlist head: ranks must be 1..={MAX_RANKS}"
            )));
        }
        let near = near.min(vocab);
        // at most 1,024 blocks, so dispatch 6 scans their counts in one pass
        let tie_block = TIE_BLOCK.max(vocab.div_ceil(1024));
        u32_of(vocab * dim, "vocab * dim")?;
        u32_of(cluster_ids.numel() + 1, "cluster_ids")?;

        let dev = weight.device();
        let words = dim / 32;
        let codes = Tensor::empty_typed(dev, &[vocab, words], DType::I32)?;
        PACK.launch_flat(
            dev,
            &[Arg::Storage(weight), Arg::Storage(&codes)],
            vocab * words,
            &[
                u32_of(vocab * words, "codes")?,
                words as u32,
                dim as u32,
                u32_of(weight.offset(), "offset")?,
            ],
        )?;
        let i32s = |n: usize| Tensor::empty_typed(dev, &[n.max(1)], DType::I32);
        Ok(ShortlistHead {
            weight: weight.clone(),
            codes,
            centroids: at_zero(centroids)?,
            cluster_ids: at_zero(cluster_ids)?,
            cluster_offsets: at_zero(cluster_offsets)?,
            successors: successors.map(at_zero).transpose()?,
            vocab,
            dim,
            clusters,
            succ_width,
            near,
            context,
            ranks,
            q: i32s(words)?,
            scores: Tensor::empty(dev, &[clusters])?,
            dist: i32s(vocab)?,
            firstpos: i32s(vocab)?,
            hist: i32s(dim + 1)?,
            plan: i32s(PLAN_ORDER + ranks)?,
            cnt: i32s(ranks)?,
            tiecnt: i32s(vocab.div_ceil(tie_block))?,
            tie_block,
            part: i32s(3 * vocab.div_ceil(SCORE_ROWS))?,
            out: i32s(4)?,
        })
    }

    /// (vocab, dim)
    pub fn shape(&self) -> [usize; 2] {
        [self.vocab, self.dim]
    }

    /// The packed sign bits of the head's rows, (vocab, dim / 32) int32.
    pub fn codes(&self) -> &Tensor {
        &self.codes
    }

    fn check_state(&self, x: &Tensor) -> Result<()> {
        if x.dtype() != DType::F32 || x.numel() != self.dim {
            return Err(Error::Shape(format!(
                "shortlist head: the state must be float32 with {} elements, got {} {:?}",
                self.dim,
                x.dtype().name(),
                x.shape()
            )));
        }
        Ok(())
    }

    /// Record the step's first `stages` dispatches.
    fn record(&self, x: &Tensor, prev: Option<usize>, stages: usize) -> Result<()> {
        self.check_state(x)?;
        let dev = x.device();
        let (v, d, c) = (self.vocab as u32, self.dim as u32, self.clusters as u32);
        let x_off = u32_of(x.offset(), "offset")?;
        let w_off = u32_of(self.weight.offset(), "offset")?;
        let context = u32_of(self.context, "context")?;
        let prev = match prev {
            Some(p) if p < self.vocab && self.succ_width > 0 => p as u32,
            _ => NONE,
        };
        let succ = self.successors.as_ref().unwrap_or(&self.cnt);
        let score_groups = self.vocab.div_ceil(SCORE_ROWS);
        let rank_groups = self.clusters.div_ceil(RANK_GROUP);
        let blocks = self.vocab.div_ceil(self.tie_block);
        let (near, ranks, tie_block) = (self.near as u32, self.ranks as u32, self.tie_block as u32);
        let launch = |k: &Kernel, args: &[Arg<'_>], groups: usize, push: &[u32]| -> Result<()> {
            k.launch(
                dev,
                args,
                [u32_of(groups, "workgroups")? * k.local_size[0], 1, 1],
                push,
            )
        };
        use Arg::Storage as S;
        for stage in 0..stages.min(STAGES) {
            match stage {
                0 => launch(
                    &CONTEXT,
                    &[
                        S(x),
                        S(&self.centroids),
                        S(&self.scores),
                        S(&self.q),
                        S(&self.hist),
                    ],
                    self.clusters.div_ceil(CONTEXT_ROWS),
                    &[c, d, x_off],
                )?,
                1 => launch(
                    &HAMMING,
                    &[
                        S(&self.q),
                        S(&self.codes),
                        S(&self.dist),
                        S(&self.firstpos),
                        S(&self.hist),
                    ],
                    self.vocab.div_ceil(HAMMING_TILE),
                    &[v, d / 32, HAMMING_TILE as u32],
                )?,
                // the ranking workgroups, then one for the k-th distance
                2 => launch(
                    &RANK,
                    &[S(&self.hist), S(&self.scores), S(&self.plan)],
                    rank_groups + 1,
                    &[d, near, c, ranks, rank_groups as u32],
                )?,
                // one workgroup a ranked cluster, then one a block of rows
                3 => launch(
                    &MARK,
                    &[
                        S(&self.plan),
                        S(&self.cluster_offsets),
                        S(&self.cluster_ids),
                        S(&self.firstpos),
                        S(&self.dist),
                        S(&self.tiecnt),
                    ],
                    self.ranks + blocks,
                    &[ranks, v, tie_block],
                )?,
                4 => launch(
                    &COUNT,
                    &[
                        S(&self.plan),
                        S(&self.cluster_offsets),
                        S(&self.cluster_ids),
                        S(&self.firstpos),
                        S(&self.cnt),
                    ],
                    self.ranks,
                    &[ranks],
                )?,
                5 => launch(
                    &CUT,
                    &[
                        S(&self.plan),
                        S(&self.cluster_offsets),
                        S(&self.cluster_ids),
                        S(&self.firstpos),
                        S(&self.cnt),
                        S(succ),
                        S(&self.dist),
                        S(&self.tiecnt),
                    ],
                    1,
                    &[
                        ranks,
                        context,
                        v,
                        self.succ_width as u32,
                        prev,
                        near,
                        tie_block,
                        blocks as u32,
                    ],
                )?,
                6 => launch(
                    &SCORE,
                    &[
                        S(x),
                        S(&self.weight),
                        S(&self.dist),
                        S(&self.firstpos),
                        S(&self.plan),
                        S(&self.part),
                    ],
                    score_groups,
                    &[v, d, x_off, w_off],
                )?,
                _ => launch(
                    &BEST,
                    &[S(&self.part), S(&self.plan), S(&self.out)],
                    1,
                    &[score_groups as u32],
                )?,
            }
        }
        Ok(())
    }

    /// The next token for state `x` (dim elements, float32, on the device),
    /// after token `prev` (for its successors; `None` for none).
    pub fn argmax(&self, x: &Tensor, prev: Option<usize>) -> Result<Choice> {
        self.record(x, prev, STAGES)?;
        let o = words_of(&self.out)?;
        if o[0] == NONE {
            return Err(Error::Invalid("shortlist head: the union was empty".into()));
        }
        Ok(Choice {
            token: o[0] as usize,
            rows: o[1] as usize,
            short: o[2] != 0,
        })
    }

    /// The union a step for `x` would score, read back (for checking).
    pub fn union(&self, x: &Tensor, prev: Option<usize>) -> Result<Union> {
        self.record(x, prev, 6)?;
        let plan = words_of(&self.plan)?;
        let (t, tie, cut, short) = (plan[0], plan[1], plan[3], plan[4] != 0);
        let dist = words_of(&self.dist)?;
        let first = words_of(&self.firstpos)?;
        let rows = (0..self.vocab as u32)
            .filter(|&r| {
                let (d, f) = (dist[r as usize], first[r as usize]);
                f < cut || d < t || (d == t && r <= tie)
            })
            .collect();
        Ok(Union { rows, short })
    }

    /// Run the step's first `stages` dispatches and wait for them (for
    /// timing where the time goes).
    pub fn profile(&self, x: &Tensor, prev: Option<usize>, stages: usize) -> Result<()> {
        self.record(x, prev, stages.min(STAGES))?;
        x.device().synchronize()
    }
}
