//! `GRILLY_SHADER_DIR` replaces a baked kernel of the same name.
//!
//! `Device::open_with(dir)` loads the baked kernels, then `dir`. The
//! directory used to be registered lazily, and the lazy register skips a
//! name that is already loaded, so a file there could add a kernel but never
//! replace one: a fixed kernel dropped into the directory was silently
//! ignored (found testing a flash-attention fix on a Colab T4).
//!
//! This compiles a `g2-add-inplace` that subtracts, and checks `add_` does.
//! It goes through the variable and the default device, as a user would;
//! it is the only test in this binary, so nothing opens the device first.

use grilly_core::{Device, Tensor};
use std::path::PathBuf;
use std::process::Command;

const SUBTRACT: &str = "#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer BufA { float a[]; };
layout(std430, set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(push_constant) uniform Params { uint total; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= total) return;
    a[i] -= b[i];
}
";

fn glslc() -> PathBuf {
    let sdk =
        PathBuf::from(std::env::var("VULKAN_SDK").expect("VULKAN_SDK (the build needs it too)"));
    ["Bin/glslc.exe", "bin/glslc", "Bin/glslc"]
        .iter()
        .map(|p| sdk.join(p))
        .find(|p| p.exists())
        .expect("glslc in the Vulkan SDK")
}

#[test]
fn a_kernel_in_the_shader_dir_replaces_the_baked_one() {
    let dir = std::env::temp_dir().join(format!("grilly-shader-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g2-add-inplace.comp");
    std::fs::write(&src, SUBTRACT).unwrap();
    let status = Command::new(glslc())
        .args(["-O", "--target-env=vulkan1.2"])
        .arg(&src)
        .arg("-o")
        .arg(dir.join("g2-add-inplace.spv"))
        .status()
        .expect("run glslc");
    assert!(status.success(), "glslc failed");
    std::fs::remove_file(&src).unwrap();

    std::env::set_var("GRILLY_SHADER_DIR", &dir); // before anything opens the device
    let dev = Device::default_device().expect("open GPU");
    let x = Tensor::from_slice(&dev, &[5.0, 7.0], &[2]).unwrap();
    x.add_(&Tensor::from_slice(&dev, &[1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let got = x.to_vec().unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        got,
        [4.0, 5.0],
        "add_ ran the baked kernel, not the shader dir's"
    );
}
