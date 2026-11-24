//! Every baked kernel fits in 48 KiB of shared memory.
//!
//! `maxComputeSharedMemorySize` is a device limit, and the devices grilly
//! runs on differ: the RX 6750 XT and lavapipe allow 64 KiB, NVIDIA's
//! Vulkan driver 48 KiB (49,152 bytes). A kernel that declares more than its
//! device allows is invalid, and NVIDIA does not refuse it: the flash
//! attention kernels, at 52-54 KiB, built and ran on AMD and lavapipe and
//! made a Colab T4 lose the device (`VK_ERROR_DEVICE_LOST`) on the first
//! prefill. Neither the development machine nor CI could see that, so this
//! reads the SPIR-V instead: it sums each kernel's Workgroup-storage
//! variables and holds them to the smallest limit grilly supports.
//!
//! Sizes are the variables' declared sizes, which is what the limit counts
//! for the arrays of scalars these kernels use.

use crate::device::EMBEDDED_KERNELS;
use std::collections::HashMap;

/// The smallest `maxComputeSharedMemorySize` among the GPUs grilly runs on
/// (NVIDIA's Vulkan driver). AMD and Intel allow 64 KiB.
const LIMIT: u32 = 48 * 1024;

const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_SPEC_CONSTANT: u32 = 50;
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;

enum Ty {
    Scalar(u32),
    Composite(u32, u32), // element type, count
    Array(u32, u32),     // element type, length constant
    Struct(Vec<u32>),
    Pointer(u32),
}

/// The bytes of Workgroup storage `spv` declares.
fn workgroup_bytes(spv: &[u8]) -> u32 {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert_eq!(words.first(), Some(&0x0723_0203), "not SPIR-V");
    let mut types = HashMap::new();
    let mut constants = HashMap::new();
    let mut shared = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        let w = &words[at..at + count];
        match op {
            OP_TYPE_BOOL => {
                types.insert(w[1], Ty::Scalar(4));
            }
            OP_TYPE_INT | OP_TYPE_FLOAT => {
                types.insert(w[1], Ty::Scalar(w[2] / 8));
            }
            OP_TYPE_VECTOR | OP_TYPE_MATRIX => {
                types.insert(w[1], Ty::Composite(w[2], w[3]));
            }
            OP_TYPE_ARRAY => {
                types.insert(w[1], Ty::Array(w[2], w[3]));
            }
            OP_TYPE_STRUCT => {
                types.insert(w[1], Ty::Struct(w[2..].to_vec()));
            }
            OP_TYPE_POINTER => {
                types.insert(w[1], Ty::Pointer(w[3]));
            }
            OP_CONSTANT | OP_SPEC_CONSTANT => {
                constants.insert(w[2], w[3]);
            }
            OP_VARIABLE if w[3] == STORAGE_WORKGROUP => shared.push(w[1]),
            _ => {}
        }
        at += count.max(1);
    }
    fn size(id: u32, types: &HashMap<u32, Ty>, constants: &HashMap<u32, u32>) -> u32 {
        match &types[&id] {
            Ty::Scalar(n) => *n,
            Ty::Composite(e, n) => size(*e, types, constants) * n,
            Ty::Array(e, len) => size(*e, types, constants) * constants[len],
            Ty::Struct(members) => members.iter().map(|m| size(*m, types, constants)).sum(),
            Ty::Pointer(to) => size(*to, types, constants),
        }
    }
    shared.iter().map(|t| size(*t, &types, &constants)).sum()
}

fn kernel(name: &str) -> &'static [u8] {
    EMBEDDED_KERNELS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no kernel {name}"))
        .1
}

#[test]
fn every_kernel_fits_in_48_kib_of_shared_memory() {
    let over: Vec<String> = EMBEDDED_KERNELS
        .iter()
        .map(|(name, spv)| (name, workgroup_bytes(spv)))
        .filter(|(_, bytes)| *bytes > LIMIT)
        .map(|(name, bytes)| format!("{name}: {bytes} bytes"))
        .collect();
    assert!(
        over.is_empty(),
        "over the {LIMIT}-byte shared memory NVIDIA allows a workgroup: {over:?}"
    );
}

/// The reader, checked against sizes worked out by hand from the GLSL, so
/// the test above cannot pass by counting nothing.
#[test]
fn the_shared_memory_reader_counts_what_the_kernels_declare() {
    // Qs, KVs: [32][129] floats; Ps: [32][33]; m_s, l_s, alpha_s: [32]
    assert_eq!(
        workgroup_bytes(kernel("g2-flash-fwd")),
        4 * (2 * 32 * 129 + 32 * 33 + 3 * 32)
    );
    // Ks, Vs: [32][129]; Qs, dOs: [8][129]; Ps, dSs: [8][33]; lse_s, drow_s: [8]
    assert_eq!(
        workgroup_bytes(kernel("g2-flash-bwd-kv")),
        4 * (2 * 32 * 129 + 2 * 8 * 129 + 2 * 8 * 33 + 2 * 8)
    );
    // Qs, dOs: [32][129]; Ks, Vs: [8][129]; dSs: [32][9]; lse_s, drow_s: [32]
    assert_eq!(
        workgroup_bytes(kernel("g2-flash-bwd-q")),
        4 * (2 * 32 * 129 + 2 * 8 * 129 + 32 * 9 + 2 * 32)
    );
}
