//! Builds v1's `grilly_core_lib` with its own CMake project, compiles the
//! extern "C" shim against v1's headers, and links both.
//!
//! Locating v1 (first match wins):
//!   1. `GRILLY_V1_DIR` — path to a grilly v1 checkout;
//!   2. a `grilly` checkout beside this repository.
//!
//! `GRILLY_CORE_LIB_DIR` points at a directory holding an already-built
//! `grilly_core_lib` and skips the CMake build. `VULKAN_SDK` must be set.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let v1 = locate_v1(&manifest);
    let sdk = PathBuf::from(
        env::var("VULKAN_SDK").expect("VULKAN_SDK must point at the LunarG Vulkan SDK"),
    );

    for var in ["GRILLY_V1_DIR", "GRILLY_CORE_LIB_DIR", "VULKAN_SDK", "CMAKE_GENERATOR"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed={}", v1.join("cpp").display());
    println!("cargo:rerun-if-changed={}", v1.join("CMakeLists.txt").display());

    let lib_dir = match env::var("GRILLY_CORE_LIB_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => build_v1(&v1),
    };

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(manifest.join("shim/gc_shim.cpp"))
        .include(manifest.join("shim"))
        .include(v1.join("cpp/include"))
        .include(v1.join("third_party/VulkanMemoryAllocator/include"))
        .include(sdk.join("Include"))
        .flag_if_supported("/EHsc")
        .compile("gc_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=grilly_core_lib");
    println!("cargo:rustc-link-search=native={}", sdk.join("Lib").display());
    println!("cargo:rustc-link-lib=dylib=vulkan-1");

    // Exposed to dependents as DEP_GRILLY_CORE_V1_DIR, so tests and tools can
    // find v1's compiled shaders without a hard-coded path.
    println!("cargo:v1_dir={}", v1.display());
    println!("cargo:rustc-env=GRILLY_SYS_V1_DIR={}", v1.display());
}

fn locate_v1(manifest: &Path) -> PathBuf {
    let candidate = match env::var("GRILLY_V1_DIR") {
        Ok(dir) => PathBuf::from(dir),
        // crates/grilly-sys -> repo root -> its parent
        Err(_) => manifest.join("../../../grilly"),
    };
    let marker = candidate.join("cpp/include/grilly/vulkan/vk_device.h");
    if !marker.exists() {
        panic!(
            "grilly v1 checkout not found at {} — set GRILLY_V1_DIR",
            candidate.display()
        );
    }
    // canonicalize() yields a `\\?\` verbatim path on Windows, which cl.exe
    // rejects as an include directory.
    let canonical = candidate.canonicalize().unwrap();
    match canonical.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(plain) => PathBuf::from(plain),
        None => canonical,
    }
}

fn build_v1(v1: &Path) -> PathBuf {
    // Always Release: a Debug build would link the debug CRT, which Rust's
    // MSVC target does not use. No injected flags: v1 is built exactly as its
    // own CMake project builds it (on MSVC that keeps /EHsc, which the shim
    // relies on to catch v1's exceptions). The generator follows the standard
    // CMAKE_GENERATOR variable when set.
    let mut cfg = cmake::Config::new(v1);
    cfg.no_default_flags(true)
        .profile("Release")
        .build_target("grilly_core_lib");
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // The cmake crate still passes CMAKE_CXX_FLAGS, which replaces MSVC's
        // defaults; restore them (the values v1's own configure produces).
        cfg.define("CMAKE_CXX_FLAGS", "/DWIN32 /D_WINDOWS /EHsc")
            .define("CMAKE_C_FLAGS", "/DWIN32 /D_WINDOWS");
    }
    let dst = cfg.build();
    let build = dst.join("build");
    for dir in [build.join("Release"), build.clone()] {
        if dir.join("grilly_core_lib.lib").exists() || dir.join("libgrilly_core_lib.a").exists() {
            return dir;
        }
    }
    panic!("grilly_core_lib not found under {}", build.display());
}
