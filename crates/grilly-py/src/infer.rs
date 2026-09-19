//! Bindings for grilly-infer. The torch-facing API, with safetensors.torch's
//! names, is python/grilly/infer/safetensors.py.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use grilly_infer::{LoadError, SafeTensorsFile};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError};
use pyo3::prelude::*;

use crate::{device, err, wrap, PyTensor};

create_exception!(
    _C,
    SafetensorError,
    PyException,
    "A file is not valid safetensors, or has no tensor of the name asked for (the safetensors package's SafetensorError)."
);

fn load_err(e: LoadError, path: Option<&Path>) -> PyErr {
    let path = path.map(|p| p.display().to_string()).unwrap_or_default();
    match e {
        LoadError::Io(e) if e.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(format!("No such file or directory: {path}"))
        }
        LoadError::Io(e) => PyOSError::new_err(format!("{path}: {e}")),
        LoadError::Format(m) => SafetensorError::new_err(m),
        LoadError::Tensor(e) => err(e),
    }
}

/// An open safetensors file: header parsed, data memory-mapped.
#[pyclass(name = "SafeFile", module = "grilly._C", frozen)]
struct PySafeFile {
    inner: SafeTensorsFile,
}

#[pymethods]
impl PySafeFile {
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<PySafeFile> {
        py.detach(|| SafeTensorsFile::open(&path))
            .map(|inner| PySafeFile { inner })
            .map_err(|e| load_err(e, Some(&path)))
    }

    /// A whole file's bytes (copied).
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PySafeFile> {
        let data = data.to_vec();
        py.detach(|| SafeTensorsFile::from_bytes(data)).map(|inner| PySafeFile { inner }).map_err(|e| load_err(e, None))
    }

    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    fn offset_keys(&self) -> Vec<String> {
        self.inner.offset_keys()
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        self.inner.metadata()
    }

    /// Upload one tensor to the default device.
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<PyTensor> {
        let dev = device()?;
        py.detach(|| self.inner.tensor(&dev, name)).map(wrap).map_err(|e| load_err(e, None))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySafeFile>()?;
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    Ok(())
}
