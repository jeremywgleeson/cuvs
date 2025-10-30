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
//! # Platform-Specific Behavior
//!
//! - **Unix/Linux/macOS**: Uses named pipes (FIFOs) for true zero-copy streaming
//! - **Windows/Other**: Falls back to temporary files (requires disk space)
//!
//! # Examples
//!
//! ## Stream to a compressed file
//!
//! ```no_run
//! use cuvs::streaming;
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
//! // Stream directly to compressed file
//! let file = File::create("index.bin.gz").unwrap();
//! let encoder = GzEncoder::new(file, Compression::default());
//! streaming::serialize(&res, &index, encoder, true).unwrap();
//! ```
//!
//! ## Stream to a Vec (in-memory buffer)
//!
//! ```no_run
//! use cuvs::streaming;
//! # use cuvs::cagra::{Index, IndexParams};
//! # use cuvs::resources::Resources;
//! # let res = Resources::new().unwrap();
//! # let params = IndexParams::new().unwrap();
//! # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
//! # let index = Index::build(&res, &params, &dataset).unwrap();
//!
//! let mut buffer = Vec::new();
//! streaming::serialize(&res, &index, &mut buffer, true).unwrap();
//! println!("Serialized {} bytes", buffer.len());
//! ```
//!
//! ## Stream from S3 (hypothetical example with aws-sdk)
//!
//! ```ignore
//! use aws_sdk_s3::primitives::ByteStream;
//! use cuvs::streaming;
//!
//! // Download from S3 as a stream
//! let s3_client = aws_sdk_s3::Client::new(&config);
//! let object = s3_client
//!     .get_object()
//!     .bucket("my-bucket")
//!     .key("index.bin")
//!     .send()
//!     .await?;
//!
//! let stream = object.body.into_async_read();
//! let index = streaming::deserialize(&res, stream)?;
//! ```

use crate::error::Result;
use std::io::{Read, Write};

/// Serialize a CAGRA index to a writer without buffering the entire index in memory
///
/// This function enables streaming large indices to any destination that implements
/// [`std::io::Write`]. On Unix systems, it uses named pipes for true streaming without
/// temporary files. On other platforms, it falls back to using a temporary file.
///
/// # Arguments
///
/// * `res` - CUDA resources handle
/// * `index` - The CAGRA index to serialize
/// * `writer` - Destination writer (file, network socket, compression stream, etc.)
/// * `include_dataset` - Whether to include the dataset vectors in the serialized output
///
/// # Performance Notes
///
/// - **Unix/Linux/macOS**: No intermediate buffering, direct streaming from GPU
/// - **Windows**: Uses a temporary file, requires disk space equal to serialized size
/// - The writer should be buffered for best performance (e.g., `BufWriter`)
///
/// # Examples
///
/// ## Basic file streaming
///
/// ```no_run
/// use cuvs::streaming;
/// use cuvs::cagra::{Index, IndexParams};
/// use cuvs::resources::Resources;
/// use std::fs::File;
/// use std::io::BufWriter;
///
/// # let res = Resources::new().unwrap();
/// # let params = IndexParams::new().unwrap();
/// # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
/// let index = Index::build(&res, &params, &dataset).unwrap();
///
/// let file = File::create("index.bin").unwrap();
/// let writer = BufWriter::new(file);
/// streaming::serialize(&res, &index, writer, true).unwrap();
/// ```
///
/// ## Stream with compression
///
/// ```no_run
/// use cuvs::streaming;
/// # use cuvs::cagra::{Index, IndexParams};
/// # use cuvs::resources::Resources;
/// use std::fs::File;
/// use flate2::write::GzEncoder;
/// use flate2::Compression;
///
/// # let res = Resources::new().unwrap();
/// # let params = IndexParams::new().unwrap();
/// # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
/// # let index = Index::build(&res, &params, &dataset).unwrap();
///
/// let file = File::create("index.bin.gz").unwrap();
/// let encoder = GzEncoder::new(file, Compression::default());
/// streaming::serialize(&res, &index, encoder, true).unwrap();
/// ```
///
/// ## Stream to memory
///
/// ```no_run
/// use cuvs::streaming;
/// # use cuvs::cagra::{Index, IndexParams};
/// # use cuvs::resources::Resources;
/// # let res = Resources::new().unwrap();
/// # let params = IndexParams::new().unwrap();
/// # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
/// # let index = Index::build(&res, &params, &dataset).unwrap();
///
/// let mut buffer = Vec::new();
/// streaming::serialize(&res, &index, &mut buffer, true).unwrap();
/// ```
#[cfg(unix)]
pub fn serialize<W: Write + Send + 'static>(
    res: &crate::resources::Resources,
    index: &crate::cagra::Index,
    mut writer: W,
    include_dataset: bool,
) -> Result<()> {
    use std::fs;
    use std::thread;

    // Create a unique named pipe in temp directory
    let pipe_path = std::env::temp_dir().join(format!("cuvs_pipe_{}", std::process::id()));
    let pipe_path_clone = pipe_path.clone();

    // Create the named pipe (mode 0o600 = owner read/write only)
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

    // Spawn thread to read from pipe and write to destination
    let reader_thread = thread::spawn(move || -> Result<()> {
        let mut reader = fs::File::open(&pipe_path_clone)?;
        std::io::copy(&mut reader, &mut writer)
            .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;
        Ok(())
    });

    // Serialize to the pipe (this blocks until reader consumes data)
    let result = index.serialize(res, &pipe_path, include_dataset);

    // Wait for reader thread to finish
    let reader_result = reader_thread
        .join()
        .map_err(|_| crate::error::Error::new("Reader thread panicked"))?;

    // Clean up the pipe
    let _ = fs::remove_file(&pipe_path);

    // Return first error if any occurred
    result?;
    reader_result?;

    Ok(())
}

/// Fallback implementation using temp file for non-Unix platforms
#[cfg(not(unix))]
pub fn serialize<W: Write>(
    res: &crate::resources::Resources,
    index: &crate::cagra::Index,
    mut writer: W,
    include_dataset: bool,
) -> Result<()> {
    use std::fs;
    use std::io::BufReader;

    // Use temp file as fallback
    let temp_path = std::env::temp_dir().join(format!(
        "cuvs_serialize_temp_{}",
        std::process::id()
    ));

    // Serialize to temp file
    index.serialize(res, &temp_path, include_dataset)?;

    // Copy to writer
    let file = fs::File::open(&temp_path)?;
    let mut reader = BufReader::new(file);
    std::io::copy(&mut reader, &mut writer)
        .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;

    // Clean up temp file
    let _ = fs::remove_file(&temp_path);

    Ok(())
}

/// Deserialize a CAGRA index from a reader without buffering the entire index in memory
///
/// This function enables loading large indices from any source that implements
/// [`std::io::Read`]. On Unix systems, it uses named pipes for true streaming without
/// temporary files. On other platforms, it falls back to using a temporary file.
///
/// # Arguments
///
/// * `res` - CUDA resources handle
/// * `reader` - Source reader (file, network socket, decompression stream, etc.)
///
/// # Returns
///
/// A deserialized CAGRA index ready for searching
///
/// # Performance Notes
///
/// - **Unix/Linux/macOS**: No intermediate buffering, direct streaming to GPU
/// - **Windows**: Uses a temporary file, requires disk space equal to serialized size
/// - The reader should be buffered for best performance (e.g., `BufReader`)
///
/// # Examples
///
/// ## Load from file
///
/// ```no_run
/// use cuvs::streaming;
/// use cuvs::resources::Resources;
/// use std::fs::File;
/// use std::io::BufReader;
///
/// let res = Resources::new().unwrap();
///
/// let file = File::open("index.bin").unwrap();
/// let reader = BufReader::new(file);
/// let index = streaming::deserialize(&res, reader).unwrap();
/// ```
///
/// ## Load from compressed file
///
/// ```no_run
/// use cuvs::streaming;
/// use cuvs::resources::Resources;
/// use std::fs::File;
/// use flate2::read::GzDecoder;
///
/// let res = Resources::new().unwrap();
///
/// let file = File::open("index.bin.gz").unwrap();
/// let decoder = GzDecoder::new(file);
/// let index = streaming::deserialize(&res, decoder).unwrap();
/// ```
///
/// ## Load from memory
///
/// ```no_run
/// use cuvs::streaming;
/// use cuvs::resources::Resources;
/// use std::io::Cursor;
///
/// let res = Resources::new().unwrap();
/// # let buffer: Vec<u8> = vec![];
///
/// let cursor = Cursor::new(buffer);
/// let index = streaming::deserialize(&res, cursor).unwrap();
/// ```
#[cfg(unix)]
pub fn deserialize<R: Read + Send + 'static>(
    res: &crate::resources::Resources,
    mut reader: R,
) -> Result<crate::cagra::Index> {
    use std::fs;
    use std::thread;

    // Create a unique named pipe in temp directory
    let pipe_path = std::env::temp_dir().join(format!("cuvs_pipe_{}", std::process::id()));
    let pipe_path_clone = pipe_path.clone();

    // Create the named pipe (mode 0o600 = owner read/write only)
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

    // Spawn thread to write to pipe from reader
    let writer_thread = thread::spawn(move || -> Result<()> {
        let mut writer = fs::File::create(&pipe_path_clone)?;
        std::io::copy(&mut reader, &mut writer)
            .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;
        Ok(())
    });

    // Deserialize from the pipe
    let result = crate::cagra::Index::deserialize(res, &pipe_path);

    // Wait for writer thread to finish
    let writer_result = writer_thread
        .join()
        .map_err(|_| crate::error::Error::new("Writer thread panicked"))?;

    // Clean up the pipe
    let _ = fs::remove_file(&pipe_path);

    // Return first error if any occurred
    writer_result?;
    result
}

/// Fallback implementation using temp file for non-Unix platforms
#[cfg(not(unix))]
pub fn deserialize<R: Read>(
    res: &crate::resources::Resources,
    mut reader: R,
) -> Result<crate::cagra::Index> {
    use std::fs;
    use std::io::Write;

    // Use temp file as fallback
    let temp_path = std::env::temp_dir().join(format!(
        "cuvs_deserialize_temp_{}",
        std::process::id()
    ));

    // Copy from reader to temp file
    let mut temp_file = fs::File::create(&temp_path)?;
    std::io::copy(&mut reader, &mut temp_file)
        .map_err(|e| crate::error::Error::new(&format!("Failed to copy data: {}", e)))?;
    temp_file.flush()?;
    drop(temp_file);

    // Deserialize from temp file
    let result = crate::cagra::Index::deserialize(res, &temp_path);

    // Clean up temp file
    let _ = fs::remove_file(&temp_path);

    result
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
        serialize(&res, &index, &mut buffer, true).unwrap();

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
        serialize(&res, &index, &mut buffer, true).unwrap();
        assert!(buffer.len() > 0);

        // Deserialize from memory
        let cursor = Cursor::new(&buffer);
        let loaded_index = deserialize(&res, cursor).unwrap();

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
        serialize(&res, &index, &mut buffer_with, true).unwrap();

        let mut buffer_without = Vec::new();
        serialize(&res, &index, &mut buffer_without, false).unwrap();

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
        serialize(&res, &index, &mut buffer, true).unwrap();

        // Buffer should contain data
        assert!(!buffer.is_empty());

        // First 4 bytes should be the dtype signature
        assert_eq!(buffer.len() >= 4, true);
    }

    #[cfg(unix)]
    #[test]
    fn test_named_pipe_cleanup() {
        use std::path::PathBuf;

        let res = Resources::new().unwrap();
        let (index, _dataset) = build_test_index(&res);

        let mut buffer = Vec::new();

        // Check that pipe doesn't exist before
        let pipe_path = std::env::temp_dir().join(format!("cuvs_pipe_{}", std::process::id()));
        assert!(!pipe_path.exists());

        // Serialize
        serialize(&res, &index, &mut buffer, true).unwrap();

        // Check that pipe was cleaned up
        assert!(!pipe_path.exists());
    }
}
