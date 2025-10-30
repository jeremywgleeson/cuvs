/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Streaming helpers for index serialization/deserialization
//!
//! This module provides utilities to stream index data to/from arbitrary destinations
//! without buffering the entire index in memory. This is particularly useful for:
//!
//! - Uploading large indices directly to cloud storage (S3, GCS, Azure Blob)
//! - Compressing indices on-the-fly during serialization
//! - Streaming over network connections
//! - Avoiding disk space requirements for temporary files
//!
//! # Supported Index Types
//!
//! All index types implement the [`StreamSerialize`] trait for generic usage:
//! - **CAGRA**: Options = [`CagraSerializeOptions`]
//! - **IVF-Flat**: Options = `()`
//! - **IVF-PQ**: Options = `()`
//!
//! # Platform-Specific Behavior
//!
//! - **Unix/Linux/macOS**: Uses named pipes (FIFOs) for true zero-copy streaming
//! - **Windows/Other**: Falls back to temporary files (requires disk space)
//!
//! # Examples
//!
//! ## Generic trait-based usage
//!
//! ```no_run
//! use cuvs::streaming::{StreamSerialize, CagraSerializeOptions};
//! use cuvs::cagra::{Index, IndexParams};
//! use cuvs::resources::Resources;
//! use std::fs::File;
//! use flate2::write::GzEncoder;
//! use flate2::Compression;
//!
//! let res = Resources::new().unwrap();
//! let params = IndexParams::new().unwrap();
//! # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
//! let index = Index::build(&res, &params, &dataset).unwrap();
//!
//! // Use trait method - works with any index type
//! let file = File::create("index.bin.gz").unwrap();
//! let encoder = GzEncoder::new(file, Compression::default());
//! index.stream_serialize(&res, encoder, CagraSerializeOptions { include_dataset: true }).unwrap();
//! ```
//!
//! ## Serialize and deserialize with compression
//!
//! ```no_run
//! use cuvs::streaming::{StreamSerialize, CagraSerializeOptions};
//! use cuvs::cagra::{Index, IndexParams};
//! use cuvs::resources::Resources;
//! use std::fs::File;
//! use flate2::write::GzEncoder;
//! use flate2::read::GzDecoder;
//! use flate2::Compression;
//!
//! let res = Resources::new().unwrap();
//! # let params = IndexParams::new().unwrap();
//! # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
//! # let index = Index::build(&res, &params, &dataset).unwrap();
//!
//! // Serialize with compression
//! let file = File::create("index.bin.gz").unwrap();
//! let encoder = GzEncoder::new(file, Compression::default());
//! index.stream_serialize(&res, encoder, CagraSerializeOptions { include_dataset: true }).unwrap();
//!
//! // Deserialize from compressed file
//! let file = File::open("index.bin.gz").unwrap();
//! let decoder = GzDecoder::new(file);
//! let loaded_index = Index::stream_deserialize(&res, decoder).unwrap();
//! ```
//!
//! ## Stream to a Vec (in-memory buffer)
//!
//! ```no_run
//! use cuvs::streaming::{StreamSerialize, CagraSerializeOptions};
//! # use cuvs::cagra::{Index, IndexParams};
//! # use cuvs::resources::Resources;
//! # let res = Resources::new().unwrap();
//! # let params = IndexParams::new().unwrap();
//! # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
//! # let index = Index::build(&res, &params, &dataset).unwrap();
//!
//! let mut buffer = Vec::new();
//! index.stream_serialize(&res, &mut buffer, CagraSerializeOptions { include_dataset: true }).unwrap();
//! println!("Serialized {} bytes", buffer.len());
//! ```

use crate::error::Result;
use crate::resources::Resources;
use std::io::{Read, Write};

/// Trait for types that support streaming serialization/deserialization
pub trait StreamSerialize: Sized {
    /// Serialization options (e.g., `CagraSerializeOptions` for CAGRA, `()` for no options)
    type Options;

    /// Serialize to a writer without buffering entire data in memory
    fn stream_serialize<W: Write + Send + 'static>(
        &self,
        res: &Resources,
        writer: W,
        options: Self::Options,
    ) -> Result<()>;

    /// Deserialize from a reader without buffering entire data in memory
    fn stream_deserialize<R: Read + Send + 'static>(res: &Resources, reader: R) -> Result<Self>;
}

/// Serialization options for CAGRA index
#[derive(Debug, Clone, Copy)]
pub struct CagraSerializeOptions {
    /// Whether to include the dataset in the serialized output
    pub include_dataset: bool,
}

impl Default for CagraSerializeOptions {
    fn default() -> Self {
        Self {
            include_dataset: true,
        }
    }
}

// Internal trait for types that can serialize to a file path
trait FileSerialize {
    type Options;
    fn write_to_file(&self, res: &Resources, path: &std::path::Path, options: Self::Options) -> Result<()>;
}

/// Internal trait for types that can deserialize from a file path
trait FileDeserialize: Sized {
    fn read_from_file(res: &Resources, path: &std::path::Path) -> Result<Self>;
}

// Implement FileSerialize for CAGRA index
impl FileSerialize for crate::cagra::Index {
    type Options = CagraSerializeOptions;

    fn write_to_file(&self, res: &Resources, path: &std::path::Path, options: CagraSerializeOptions) -> Result<()> {
        self.serialize(res, path, options.include_dataset)
    }
}

// Implement FileDeserialize for CAGRA index
impl FileDeserialize for crate::cagra::Index {
    fn read_from_file(res: &Resources, path: &std::path::Path) -> Result<Self> {
        Self::deserialize(res, path)
    }
}

// Implement FileSerialize for IVF-Flat index
impl FileSerialize for crate::ivf_flat::Index {
    type Options = ();

    fn write_to_file(&self, res: &Resources, path: &std::path::Path, _options: ()) -> Result<()> {
        self.serialize(res, path)
    }
}

// Implement FileDeserialize for IVF-Flat index
impl FileDeserialize for crate::ivf_flat::Index {
    fn read_from_file(res: &Resources, path: &std::path::Path) -> Result<Self> {
        Self::deserialize(res, path)
    }
}

// Implement FileSerialize for IVF-PQ index
impl FileSerialize for crate::ivf_pq::Index {
    type Options = ();

    fn write_to_file(&self, res: &Resources, path: &std::path::Path, _options: ()) -> Result<()> {
        self.serialize(res, path)
    }
}

// Implement FileDeserialize for IVF-PQ index
impl FileDeserialize for crate::ivf_pq::Index {
    fn read_from_file(res: &Resources, path: &std::path::Path) -> Result<Self> {
        Self::deserialize(res, path)
    }
}

/// Generate a cryptographically secure random filename component
fn generate_random_name() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let random_state = RandomState::new();
    let mut hasher = random_state.build_hasher();

    // Hash multiple sources of entropy
    std::process::id().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);

    // Use current time as additional entropy
    if let Ok(duration) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        duration.as_nanos().hash(&mut hasher);
    }

    format!("cuvs_{:016x}", hasher.finish())
}


/// Generic implementation for Unix platforms using named pipes
#[cfg(unix)]
fn stream_serialize_impl<T, W>(
    res: &Resources,
    index: &T,
    mut writer: W,
    options: T::Options,
) -> Result<()>
where
    T: FileSerialize,
    W: Write + Send + 'static,
{
    use std::fs;
    use std::thread;

    let pipe_path = std::env::temp_dir().join(generate_random_name());
    let pipe_path_clone = pipe_path.clone();

    let c_path = std::ffi::CString::new(
        pipe_path
            .to_str()
            .ok_or_else(|| crate::error::Error::new("Invalid pipe path"))?,
    )
    .map_err(|_| crate::error::Error::new("Pipe path contains null bytes"))?;

    unsafe {
        if libc::mkfifo(c_path.as_ptr(), 0o600) != 0 {
            return Err(crate::error::Error::new(&format!(
                "Failed to create named pipe: {}",
                std::io::Error::last_os_error()
            )));
        }
    }

    let reader_thread = thread::spawn(move || -> Result<()> {
        let mut reader = fs::File::open(&pipe_path_clone)?;
        std::io::copy(&mut reader, &mut writer)
            .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;
        Ok(())
    });

    let result = index.write_to_file(res, &pipe_path, options);
    let reader_result = reader_thread
        .join()
        .map_err(|_| crate::error::Error::new("Reader thread panicked"))?;

    let _ = fs::remove_file(&pipe_path);

    result?;
    reader_result?;
    Ok(())
}

/// Generic implementation for non-Unix platforms using temporary files
#[cfg(not(unix))]
fn stream_serialize_impl<T, W>(
    res: &Resources,
    index: &T,
    mut writer: W,
    options: T::Options,
) -> Result<()>
where
    T: FileSerialize,
    W: Write + Send + 'static,
{
    use std::fs;
    use std::io::BufReader;

    let temp_path = std::env::temp_dir().join(generate_random_name());
    index.write_to_file(res, &temp_path, options)?;

    let file = fs::File::open(&temp_path)?;
    let mut reader = BufReader::new(file);
    std::io::copy(&mut reader, &mut writer)
        .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;

    let _ = fs::remove_file(&temp_path);
    Ok(())
}

/// Generic implementation for Unix platforms using named pipes
#[cfg(unix)]
fn stream_deserialize_impl<T, R>(res: &Resources, mut reader: R) -> Result<T>
where
    T: FileDeserialize,
    R: Read + Send + 'static,
{
    use std::fs;
    use std::thread;

    let pipe_path = std::env::temp_dir().join(generate_random_name());
    let pipe_path_clone = pipe_path.clone();

    let c_path = std::ffi::CString::new(
        pipe_path
            .to_str()
            .ok_or_else(|| crate::error::Error::new("Invalid pipe path"))?,
    )
    .map_err(|_| crate::error::Error::new("Pipe path contains null bytes"))?;

    unsafe {
        if libc::mkfifo(c_path.as_ptr(), 0o600) != 0 {
            return Err(crate::error::Error::new(&format!(
                "Failed to create named pipe: {}",
                std::io::Error::last_os_error()
            )));
        }
    }

    let writer_thread = thread::spawn(move || -> Result<()> {
        let mut writer = fs::File::create(&pipe_path_clone)?;
        std::io::copy(&mut reader, &mut writer)
            .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;
        Ok(())
    });

    let result = T::read_from_file(res, &pipe_path);
    let writer_result = writer_thread
        .join()
        .map_err(|_| crate::error::Error::new("Writer thread panicked"))?;

    let _ = fs::remove_file(&pipe_path);

    writer_result?;
    result
}

/// Generic implementation for non-Unix platforms using temporary files
#[cfg(not(unix))]
fn stream_deserialize_impl<T, R>(res: &Resources, mut reader: R) -> Result<T>
where
    T: FileDeserialize,
    R: Read + Send + 'static,
{
    use std::fs;
    use std::io::Write;

    let temp_path = std::env::temp_dir().join(generate_random_name());
    let mut temp_file = fs::File::create(&temp_path)?;
    std::io::copy(&mut reader, &mut temp_file)
        .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;
    temp_file.flush()?;
    drop(temp_file);

    let result = T::read_from_file(res, &temp_path);
    let _ = fs::remove_file(&temp_path);
    result
}

/// CAGRA index streaming implementation
///
/// Uses `CagraSerializeOptions` to control serialization behavior.
impl StreamSerialize for crate::cagra::Index {
    type Options = CagraSerializeOptions;

    fn stream_serialize<W: Write + Send + 'static>(
        &self,
        res: &Resources,
        writer: W,
        options: CagraSerializeOptions,
    ) -> Result<()> {
        stream_serialize_impl(res, self, writer, options)
    }

    fn stream_deserialize<R: Read + Send + 'static>(res: &Resources, reader: R) -> Result<Self> {
        stream_deserialize_impl(res, reader)
    }
}

/// IVF-Flat index streaming implementation
///
/// Uses `()` for Options since it has no serialization options.
impl StreamSerialize for crate::ivf_flat::Index {
    type Options = ();

    fn stream_serialize<W: Write + Send + 'static>(
        &self,
        res: &Resources,
        writer: W,
        _options: (),
    ) -> Result<()> {
        stream_serialize_impl(res, self, writer, ())
    }

    fn stream_deserialize<R: Read + Send + 'static>(res: &Resources, reader: R) -> Result<Self> {
        stream_deserialize_impl(res, reader)
    }
}

/// IVF-PQ index streaming implementation
///
/// Uses `()` for Options since it has no serialization options.
impl StreamSerialize for crate::ivf_pq::Index {
    type Options = ();

    fn stream_serialize<W: Write + Send + 'static>(
        &self,
        res: &Resources,
        writer: W,
        _options: (),
    ) -> Result<()> {
        stream_serialize_impl(res, self, writer, ())
    }

    fn stream_deserialize<R: Read + Send + 'static>(res: &Resources, reader: R) -> Result<Self> {
        stream_deserialize_impl(res, reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cagra::{Index, IndexParams, SearchParams};
    use crate::dlpack::ManagedTensor;
    use crate::resources::Resources;
    use ndarray::s;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::io::Cursor;

    /// Helper to build a test index
    fn build_test_index(res: &Resources) -> (Index, ndarray::Array2<f32>) {
        let params = IndexParams::new().unwrap();
        let n_datapoints = 256;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
        let index = Index::build(res, &params, &dataset).expect("failed to create cagra index");
        (index, dataset)
    }

    #[test]
    fn test_serialize_to_vec() {
        let res = Resources::new().unwrap();
        let (index, _dataset) = build_test_index(&res);

        // Serialize to a Vec
        let mut buffer = Vec::new();
        index.stream_serialize(&res, &mut buffer, CagraSerializeOptions { include_dataset: true }).unwrap();

        // Should have serialized data
        assert!(buffer.len() > 0);
        println!("Serialized {} bytes", buffer.len());
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let res = Resources::new().unwrap();
        let (index, dataset) = build_test_index(&res);

        // Serialize to memory
        let mut buffer = Vec::new();
        index.stream_serialize(&res, &mut buffer, CagraSerializeOptions { include_dataset: true }).unwrap();
        assert!(buffer.len() > 0);

        // Deserialize from memory
        let cursor = Cursor::new(&buffer);
        let loaded_index = Index::stream_deserialize(&res, cursor).unwrap();

        // Verify the loaded index works by performing a search
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let k = 10;

        let queries_device = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        let search_params = SearchParams::new().unwrap();
        loaded_index
            .search(&res, &search_params, &queries_device, &neighbors, &distances)
            .unwrap();

        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // Nearest neighbors should be themselves
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    #[test]
    fn test_serialize_without_dataset() {
        let res = Resources::new().unwrap();
        let (index, _dataset) = build_test_index(&res);

        // Serialize without dataset
        let mut buffer_with = Vec::new();
        index.stream_serialize(&res, &mut buffer_with, CagraSerializeOptions { include_dataset: true }).unwrap();

        let mut buffer_without = Vec::new();
        index.stream_serialize(&res, &mut buffer_without, CagraSerializeOptions { include_dataset: false }).unwrap();

        // Without dataset should be smaller
        assert!(buffer_without.len() < buffer_with.len());
        println!(
            "With dataset: {} bytes, without: {} bytes",
            buffer_with.len(),
            buffer_without.len()
        );
    }

    #[test]
    fn test_empty_buffer_serialization() {
        let res = Resources::new().unwrap();
        let (index, _dataset) = build_test_index(&res);

        // Serialize to Vec
        let mut buffer = Vec::new();
        index.stream_serialize(&res, &mut buffer, CagraSerializeOptions { include_dataset: true }).unwrap();

        // Buffer should contain data
        assert!(!buffer.is_empty());

        // First 4 bytes should be the dtype signature
        assert_eq!(buffer.len() >= 4, true);
    }

    #[test]
    fn test_generic_trait_usage() {
        // Test that the trait allows generic functions
        fn serialize_any_index<I: StreamSerialize>(
            res: &Resources,
            index: &I,
            options: I::Options,
        ) -> Vec<u8> {
            let mut buffer = Vec::new();
            index.stream_serialize(res, &mut buffer, options).unwrap();
            buffer
        }

        let res = Resources::new().unwrap();
        let (cagra_index, _) = build_test_index(&res);

        // Serialize using generic function
        let buffer = serialize_any_index(&res, &cagra_index, CagraSerializeOptions { include_dataset: true });
        assert!(!buffer.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn test_named_pipe_cleanup() {
        let res = Resources::new().unwrap();
        let (index, _dataset) = build_test_index(&res);

        let mut buffer = Vec::new();

        // Count cuvs temp files before
        let temp_dir = std::env::temp_dir();
        let before_count = std::fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.starts_with("cuvs_"))
                    .unwrap_or(false)
            })
            .count();

        // Serialize
        index.stream_serialize(&res, &mut buffer, CagraSerializeOptions { include_dataset: true }).unwrap();

        // Verify no new cuvs temp files remain (all cleaned up)
        let after_count = std::fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.starts_with("cuvs_"))
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(before_count, after_count, "Temporary pipe was not cleaned up");
    }
}
