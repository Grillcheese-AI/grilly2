//! Compiles grilly2's Vulkan layer and the `extern "C"` shim over it, and
//! links them against Vulkan and shaderc. No CMake, no second project, no
//! grilly v1.
//!
//! `vulkan/` holds the device, buffer pool, pipeline cache and command
//! batch — the four classes the shim names, brought over from v1 when the
//! link to it was cut. They depend on nothing but `<vulkan/vulkan.h>`,
//! VulkanMemoryAllocator and the standard library.
//!
//! Two things have to be found:
//!   - `VULKAN_SDK` — the LunarG SDK, for the headers, `vulkan-1` and
//!     `shaderc_combined` (the shader JIT behind `gc_compile_shader`);
//!   - VulkanMemoryAllocator's single header, from `VMA_INCLUDE_DIR` if
//!     set, otherwise the submodule under `third_party/`.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk = PathBuf::from(
        env::var("VULKAN_SDK").expect("VULKAN_SDK must point at the LunarG Vulkan SDK"),
    );
    let vma = locate_vma(&manifest);

    for var in ["VULKAN_SDK", "VMA_INCLUDE_DIR"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=vulkan");

    let vulkan = manifest.join("vulkan");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(manifest.join("shim/gc_shim.cpp"))
        .file(vulkan.join("device.cpp"))
        .file(vulkan.join("buffer_pool.cpp"))
        .file(vulkan.join("pipeline_cache.cpp"))
        .file(vulkan.join("command_batch.cpp"))
        .include(manifest.join("shim"))
        .include(vulkan.join("include"))
        .include(&vma)
        .include(sdk.join("Include"))
        // the shim catches the Vulkan layer's exceptions, so unwinding has
        // to be on; MSVC defaults it off for C++ without this
        .flag_if_supported("/EHsc")
        .compile("grilly_vulkan");

    println!("cargo:rustc-link-search=native={}", sdk.join("Lib").display());
    println!("cargo:rustc-link-lib=dylib=vulkan-1");
    println!("cargo:rustc-link-lib=static=shaderc_combined");
}

/// VulkanMemoryAllocator's include directory.
///
/// The submodule is the default so a `--recursive` clone builds with
/// nothing installed; `VMA_INCLUDE_DIR` overrides it for a distribution
/// that already ships the header.
fn locate_vma(manifest: &Path) -> PathBuf {
    let candidate = match env::var("VMA_INCLUDE_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => manifest.join("third_party/VulkanMemoryAllocator/include"),
    };
    if !candidate.join("vk_mem_alloc.h").exists() {
        panic!(
            "vk_mem_alloc.h not found in {} — run\n    \
             git submodule update --init --recursive\n\
             or point VMA_INCLUDE_DIR at a VulkanMemoryAllocator include directory",
            candidate.display()
        );
    }
    plain(candidate)
}

/// A path MSVC will accept as an include directory.
///
/// `canonicalize()` yields a `\\?\` verbatim path on Windows, which
/// `cl.exe` rejects.
fn plain(path: PathBuf) -> PathBuf {
    let canonical = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => return path,
    };
    match canonical.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(p) => PathBuf::from(p),
        None => canonical,
    }
}
