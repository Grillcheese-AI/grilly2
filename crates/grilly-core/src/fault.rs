//! Index bounds checked on the device, reported at the next host read.
//!
//! An index op has to know its indices are in range. Reading them back to
//! find out costs a readback — which ends the command batch — and, worse,
//! makes the op impossible to capture: a replay re-issues commands without
//! running the host again, so anything the host decided cannot be
//! reproduced and the capture is refused. That kept every model with an
//! embedding out of `grilly.graphed`.
//!
//! torch does not ask the host either. CUDA asserts on the device and the
//! error surfaces at the next synchronize, which is why a bad index there
//! reports at a line that looks unrelated. This is the same bargain, made
//! explicit: `g2-index-fault` records the first bad value in a small
//! host-visible buffer, and the next time the host reads anything the
//! fault is raised and the record cleared.
//!
//! The record is one per process, like the device. It is
//! [`MemClass::Readback`] so reading it is a mapped copy rather than a
//! transfer, and it is only read at all when an index op has armed it
//! since the last check — so a program that never indexes never pays.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::device::MemClass;
use crate::{Arg, DType, Device, Error, Kernel, Result, Tensor};

/// Bindings: index, record. Push: elements, index offset, size,
/// allow_negative, kind, dim.
const CHECK: Kernel = Kernel { shader: "g2-index-fault", bindings: 2, push_words: 6, local_size: [256, 1, 1] };

pub const KERNELS: &[Kernel] = &[CHECK];

/// Which op armed the check, so the message can be the one torch gives.
/// Zero means "no fault", so the codes start at one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexOp {
    Gather = 1,
    IndexSelect = 2,
    Index = 3,
    Embedding = 4,
    Scatter = 5,
}

impl IndexOp {
    fn from_code(code: i32) -> IndexOp {
        match code {
            1 => IndexOp::Gather,
            2 => IndexOp::IndexSelect,
            4 => IndexOp::Embedding,
            5 => IndexOp::Scatter,
            _ => IndexOp::Index,
        }
    }

    fn name(self) -> &'static str {
        match self {
            IndexOp::Gather => "gather",
            IndexOp::IndexSelect => "index_select",
            IndexOp::Index => "indexing with a tensor",
            IndexOp::Embedding => "embedding",
            IndexOp::Scatter => "scatter",
        }
    }

    /// torch's wording for this op, which downstream code matches on, plus
    /// a note saying where it came from.
    ///
    /// The note earns its place: the check runs on the device, so the error
    /// arrives at whatever read next rather than at the line that indexed.
    /// Without it the report looks like a fault in unrelated code.
    fn message(self, value: i32, size: i32, dim: i32) -> Error {
        let where_from = format!(
            " (recorded by {}; grilly checks index bounds on the device and reports at the next              read, as torch does on CUDA, so this is not the line that indexed)",
            self.name()
        );
        match self {
            IndexOp::Embedding => Error::Index(format!("index out of range in self{where_from}")),
            IndexOp::IndexSelect => Error::Invalid(format!(
                "INDICES element is out of DATA bounds, id={value} axis_dim={size}{where_from}"
            )),
            IndexOp::Index => Error::Index(format!(
                "index {value} is out of bounds for dimension {dim} with size {size}{where_from}"
            )),
            IndexOp::Gather | IndexOp::Scatter => Error::Invalid(format!(
                "index {value} is out of bounds for dimension {dim} with size {size}{where_from}"
            )),
        }
    }
}

static RECORD: OnceLock<Mutex<Option<Tensor>>> = OnceLock::new();
/// Whether an index op has run since the last check. Reading the record
/// costs a small mapped copy, and this keeps it to programs that index.
static ARMED: AtomicBool = AtomicBool::new(false);

fn record(device: &Device) -> Result<Tensor> {
    let slot = RECORD.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = guard.as_ref() {
        return Ok(t.clone());
    }
    let t = Tensor::empty_in(device, &[4], DType::I32, MemClass::Readback)?;
    t.fill_(0.0)?;
    *guard = Some(t.clone());
    Ok(t)
}

/// Record a fault if any of `index`'s values is outside `[lo, size)`.
///
/// Issues one dispatch and returns. Nothing is read, so this is safe
/// inside a capture and the check replays with the commands it guards.
pub fn check_indices(
    index: &Tensor,
    size: usize,
    allow_negative: bool,
    op: IndexOp,
    dim: usize,
) -> Result<()> {
    if index.dtype() != DType::I64 {
        return Err(Error::Invalid(format!(
            "index bounds check: indices must be int64, got {}",
            index.dtype().name()
        )));
    }
    let n = index.numel();
    if n == 0 {
        return Ok(());
    }
    let device = index.device();
    let record = record(device)?;
    ARMED.store(true, Ordering::Release);
    CHECK.launch_flat(
        device,
        &[Arg::Storage(index), Arg::Tensor(&record)],
        n,
        &[
            n as u32,
            index.offset() as u32,
            u32::try_from(size).map_err(|_| Error::Shape(format!("dimension {size} exceeds u32")))?,
            u32::from(allow_negative),
            op as u32,
            u32::try_from(dim).unwrap_or(0),
        ],
    )
}

/// Raise if an index op recorded a fault, and clear the record.
///
/// Called after a host read or a synchronize, when the batch is already
/// empty — so the read here adds no submit. Does nothing during a capture,
/// where reading is refused; the fault stays armed and surfaces at the
/// first host read after the capture closes.
pub fn take(device: &Device) -> Result<()> {
    if device.capturing() || !ARMED.swap(false, Ordering::AcqRel) {
        return Ok(());
    }
    let slot = RECORD.get_or_init(|| Mutex::new(None));
    let tensor = {
        let guard = slot.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(t) => t.clone(),
            None => return Ok(()),
        }
    };
    let bytes = tensor.to_bytes()?;
    let word = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let kind = word(0);
    if kind == 0 {
        return Ok(());
    }
    tensor.fill_(0.0)?;
    Err(IndexOp::from_code(kind).message(word(1), word(2), word(3)))
}
