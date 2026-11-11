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
    // grilly-sys builds before grilly-core, so this is the message someone
    // sees after `pip install grilly2` falls back to the sdist
    let sdk = PathBuf::from(env::var("VULKAN_SDK").unwrap_or_else(|_| {
        panic!(
            "\n\
             grilly2 is being built from source, which needs the LunarG Vulkan SDK \n\
             (headers, glslc, shaderc). VULKAN_SDK is not set.\n\n\
             If you did not mean to build from source, there is probably no prebuilt \n\
             wheel for this platform yet — https://github.com/Grillcheese-AI/grilly2/issues\n\n\
             To build: install the SDK from https://vulkan.lunarg.com/sdk/home and set \n\
             VULKAN_SDK. Running grilly2 needs only your GPU driver.\n"
        )
    }));
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
        .include(sdk_dir(&sdk, &["Include", "include"], "vulkan/vulkan.h"))
        // the shim catches the Vulkan layer's exceptions, so unwinding has
        // to be on; MSVC defaults it off for C++ without this
        .flag_if_supported("/EHsc")
        .compile("grilly_vulkan");

    // The Windows SDK names the loader `vulkan-1.lib`; everywhere else it
    // is `libvulkan.so`. Only the import library is linked: at run time the
    // loader is the one the GPU driver installed (see release.yml).
    let windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let (loader, shaderc) = if windows {
        ("vulkan-1", "shaderc_combined.lib")
    } else {
        ("vulkan", "libshaderc_combined.a")
    };
    let lib = sdk_dir(&sdk, &["Lib", "lib", "lib64"], shaderc);
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib={loader}");
    println!("cargo:rustc-link-lib=static=shaderc_combined");
}

/// The first of `names` under the SDK that holds `marker`.
///
/// The Windows SDK spells its directories `Include` and `Lib`, the Linux
/// tarball `include` and `lib`, and Linux file systems are case-sensitive.
fn sdk_dir(sdk: &Path, names: &[&str], marker: &str) -> PathBuf {
    names
        .iter()
        .map(|n| sdk.join(n))
        .find(|d| d.join(marker).exists())
        .unwrap_or_else(|| {
            panic!(
                "{marker} not found under VULKAN_SDK={} (looked in {names:?}) — \
                 is VULKAN_SDK the SDK root for this platform? On Linux that is \
                 the x86_64/ directory inside the tarball.",
                sdk.display()
            )
        })
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
