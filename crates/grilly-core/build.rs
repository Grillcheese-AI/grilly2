//! Compiles grilly2's own kernels (shaders/*.glsl) to SPIR-V with glslc from
//! the Vulkan SDK. The .spv files go to OUT_DIR and are never committed; the
//! directory is exposed to the crate as GRILLY_CORE_SPV_DIR.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders = manifest.join("shaders");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("spv");
    std::fs::create_dir_all(&out).unwrap();

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    let sdk = PathBuf::from(env::var("VULKAN_SDK").expect("VULKAN_SDK must point at the LunarG Vulkan SDK"));
    let glslc = ["Bin/glslc.exe", "bin/glslc", "Bin/glslc"]
        .iter()
        .map(|p| sdk.join(p))
        .find(|p| p.exists())
        .expect("glslc not found in the Vulkan SDK");

    for entry in std::fs::read_dir(&shaders).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
        // g2-common.glsl is only #include-d.
        if path.extension().is_none_or(|e| e != "glsl") || name == "g2-common" {
            continue;
        }
        let status = Command::new(&glslc)
            .args(["-fshader-stage=compute", "-O", "--target-env=vulkan1.2", "-I"])
            .arg(&shaders)
            .arg(&path)
            .arg("-o")
            .arg(out.join(format!("{name}.spv")))
            .status()
            .expect("failed to run glslc");
        assert!(status.success(), "glslc failed on {}", path.display());
    }
    println!("cargo:rustc-env=GRILLY_CORE_SPV_DIR={}", out.display());
}
