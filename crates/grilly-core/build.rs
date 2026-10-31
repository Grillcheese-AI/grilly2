//! Compiles grilly2's own kernels (shaders/*.glsl) to SPIR-V with glslc from
//! the Vulkan SDK, then writes `kernels.rs` — an `include_bytes!` table that
//! puts every blob **inside the binary**.
//!
//! The table is what makes grilly2 installable. The .spv files land in
//! OUT_DIR, whose absolute path exists only on the machine that built the
//! crate — `target/release/build/grilly-core-<hash>/out/spv`, cargo hash and
//! all. An earlier version handed that path to the crate as
//! `GRILLY_CORE_SPV_DIR` and read the directory at device bring-up, which
//! meant the extension was **not relocatable**: moving the repo broke the
//! installed module, and a wheel built on a CI runner would have failed at
//! `import grilly` on every user's machine, looking for a directory that
//! never existed there.
//!
//! `include_bytes!` bakes the *contents*, so the shipped binary holds no
//! path at all. 173 kernels, 973 KB. `GRILLY_SHADER_DIR` still overrides
//! them at runtime, because it is loaded afterwards.

use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders = manifest.join("shaders");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("spv");
    std::fs::create_dir_all(&out).unwrap();

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    // A user-facing string, not a developer note: pip falls back to the
    // sdist on any platform without a wheel, so this is what someone sees
    // after `pip install grilly2` on an unsupported platform.
    let sdk = PathBuf::from(env::var("VULKAN_SDK").unwrap_or_else(|_| {
        panic!(
            "\n\
             grilly2 is being built from source, which needs the LunarG Vulkan SDK \n\
             for its headers, glslc and shaderc. VULKAN_SDK is not set.\n\n\
             If you did not mean to build from source, there is probably no prebuilt \n\
             wheel for this platform yet — https://github.com/Grillcheese-AI/grilly2/issues\n\n\
             To build: install the SDK from https://vulkan.lunarg.com/sdk/home and make \n\
             sure VULKAN_SDK is set (the installer does this; a new shell may be needed).\n\n\
             Note this is a BUILD dependency only. Running grilly2 needs just your GPU \n\
             driver, which already ships the Vulkan loader.\n"
        )
    }));
    let glslc = ["Bin/glslc.exe", "bin/glslc", "Bin/glslc"]
        .iter()
        .map(|p| sdk.join(p))
        .find(|p| p.exists())
        .expect("glslc not found in the Vulkan SDK");

    let mut names = Vec::new();
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
        names.push(name);
    }
    // sorted, so the table's order — and therefore which of two kernels of
    // one name wins — does not depend on the order the filesystem hands
    // back directory entries
    names.sort();
    assert!(!names.is_empty(), "no kernels compiled from {}", shaders.display());

    let mut table = String::from(
        "/// Every kernel, baked into the binary by build.rs. `(name, SPIR-V)`.\n\
         pub(crate) static EMBEDDED_KERNELS: &[(&str, &[u8])] = &[\n",
    );
    for name in &names {
        // the path is a build-machine path, and that is fine: `include_bytes!`
        // resolves it now and stores the bytes, so none of it reaches the binary
        writeln!(
            table,
            "    ({:?}, include_bytes!({:?})),",
            name,
            out.join(format!("{name}.spv"))
        )
        .unwrap();
    }
    table.push_str("];\n");
    std::fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("kernels.rs"), table).unwrap();
}
