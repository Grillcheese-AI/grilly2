//! Capture a sequence of GPU commands once, then replay it.
//!
//! The Vulkan analogue of CUDA graphs, at the shim boundary. A decode step
//! is the same commands on the same buffers every token; what changes is
//! the bytes inside them. Running the Python again to rediscover that costs
//! far more than the commands do — measured on this card, about 10 µs of
//! host time per dispatch from Python against 1.1 µs to record one at the
//! shim (docs/submission.md). A replay skips the model's Python, grilly's
//! Python, PyO3 and every shape check, and re-issues the commands.
//!
//! **The contract.** A replay re-runs commands over the *same buffers*, so
//! it is correct exactly when the step is a pure function of the bytes in
//! those buffers:
//!
//! - write the step's new inputs into the tensors it was captured with,
//!   in place, then replay;
//! - every intermediate the capture allocated is retained by the graph and
//!   written again by the replay;
//! - shapes, and which kernel runs, are fixed at capture. A different shape
//!   needs a different graph — `grilly.CommandGraph` keys them by the
//!   signature the caller gives.
//!
//! **Capture is device-wide.** grilly has one queue and one device, so a
//! capture sweeps up whatever else that device is asked to do while it is
//! open — as CUDA's stream capture does for its stream. Capture from the
//! thread that will replay, with nothing else running on the GPU.
//!
//! A host read during capture is refused: it would mean the step's control
//! flow depends on data, which a replay cannot reproduce. A host *write* is
//! refused for the mirror reason — a replay would write the same bytes
//! again, so a value that changes per step has to live in a tensor the
//! caller updates between replays.

use std::ffi::CString;
use std::sync::Arc;

use grilly_sys as sys;

use crate::device::{Buffer, Device};
use crate::error::{Error, Result};

/// One recorded command, with everything the shim needs to re-issue it.
pub(crate) enum Command {
    Dispatch { shader: CString, bindings: Vec<sys::gc_binding>, groups: [u32; 3], push: Vec<u32> },
    Copy { src: sys::gc_buf, src_offset: u64, dst: sys::gc_buf, dst_offset: u64, bytes: u64 },
    Fill { dst: sys::gc_buf, offset: u64, bytes: u64, word: u32 },
}

/// Commands recorded so far, and the buffers they reference.
#[derive(Default)]
pub struct Recorder {
    pub(crate) commands: Vec<Command>,
    /// Keeps every referenced buffer alive: a replay re-issues commands that
    /// name these handles, so the graph must outlive the tensors that made
    /// them.
    pub(crate) retained: Vec<Arc<Buffer>>,
}

/// A captured command sequence, replayable without running Rust's op layer
/// or Python again.
pub struct CommandGraph {
    device: Device,
    commands: Vec<Command>,
    _retained: Vec<Arc<Buffer>>,
}

impl CommandGraph {
    pub fn new(device: Device, recorder: Recorder) -> CommandGraph {
        CommandGraph { device, commands: recorder.commands, _retained: recorder.retained }
    }

    /// Commands in the graph.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Re-issue every command, in order, each behind the same full barrier a
    /// freshly recorded one gets.
    pub fn replay(&self) -> Result<()> {
        for command in &self.commands {
            match command {
                Command::Dispatch { shader, bindings, groups, push } => {
                    self.device.record(shader, bindings, *groups, push)?;
                }
                Command::Copy { src, src_offset, dst, dst_offset, bytes } => {
                    self.device.record_copy_raw(*src, *src_offset, *dst, *dst_offset, *bytes)?;
                }
                Command::Fill { dst, offset, bytes, word } => {
                    self.device.record_fill_raw(*dst, *offset, *bytes, *word)?;
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for CommandGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CommandGraph({} commands)", self.commands.len())
    }
}

/// Run `body` with capture on, and return what it recorded.
///
/// The body also *runs*: capturing is recording what a real step did, so the
/// first token is computed as usual and later ones replay it.
pub fn capture<T>(device: &Device, body: impl FnOnce() -> Result<T>) -> Result<(T, CommandGraph)> {
    device.begin_capture()?;
    let value = match body() {
        Ok(value) => value,
        Err(e) => {
            let _ = device.end_capture();
            return Err(e);
        }
    };
    let recorder = device.end_capture()?;
    if recorder.commands.is_empty() {
        return Err(Error::Invalid(
            "nothing was captured: the callable recorded no GPU commands, so there is nothing to replay".into(),
        ));
    }
    Ok((value, CommandGraph::new(device.clone(), recorder)))
}
