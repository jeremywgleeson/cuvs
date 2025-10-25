/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Filters for approximate nearest neighbor search
//!
//! This module provides filtering functionality for ANN search operations,
//! allowing you to exclude certain vectors from search results.
//!
//! # Filter Types
//!
//! - **No Filter**: Default behavior, includes all vectors
//! - **Bitset**: Global filter applied to all queries
//! - **Bitmap**: Per-query filter for batch operations
//!
//! # Example
//!
//! ```no_run
//! use cuvs::filters::{Filter, Bitset, Bitmap};
//! use cuvs::{Resources, ManagedTensor};
//! use ndarray::Array1;
//!
//! let res = Resources::new().unwrap();
//!
//! // Create a bitset filter that excludes even-indexed vectors
//! let n_samples = 1000;
//! let bitset_size = (n_samples + 31) / 32;
//! let mut bitset_data = Array1::<u32>::zeros(bitset_size);
//! for i in 0..bitset_size {
//!     // 0xAAAAAAAA = binary 10101010... (filters even indices)
//!     bitset_data[i] = 0xAAAAAAAA;
//! }
//! let bitset_tensor = ManagedTensor::from(&bitset_data).to_device(&res).unwrap();
//! let filter = Bitset::new(&bitset_tensor, n_samples);
//!
//! // Use this filter with any search operation
//! // index.search(&res, &search_params, &queries, &neighbors, &distances, &filter);
//! ```

use crate::dlpack::ManagedTensor;

/// Filter type for ANN search operations
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// No filtering - all vectors are considered
    NoFilter,
    /// Bitset filter - global filter for all queries
    Bitset,
    /// Bitmap filter - per-query filter for batch operations
    Bitmap,
}

impl From<FilterType> for ffi::cuvsFilterType {
    fn from(filter_type: FilterType) -> Self {
        match filter_type {
            FilterType::NoFilter => ffi::cuvsFilterType::NO_FILTER,
            FilterType::Bitset => ffi::cuvsFilterType::BITSET,
            FilterType::Bitmap => ffi::cuvsFilterType::BITMAP,
        }
    }
}

/// Base trait for all filter types
pub trait Filter {
    /// Convert this filter into a C FFI filter struct
    fn into_ffi(&self) -> ffi::cuvsFilter;
}

/// No filter - includes all vectors in search results
///
/// This is the default behavior when no filter is specified.
#[derive(Debug)]
pub struct NoFilter;

impl Filter for NoFilter {
    fn into_ffi(&self) -> ffi::cuvsFilter {
        ffi::cuvsFilter {
            addr: 0,
            type_: ffi::cuvsFilterType::NO_FILTER,
        }
    }
}

/// Bitset filter - applies the same filter to all queries
///
/// A bitset is a compact representation where each bit indicates whether
/// a vector should be included (1) or excluded (0) from search results.
///
/// # Layout
///
/// - Shape: `[(n_samples + 31) / 32]` (uint32 elements)
/// - Each bit represents one vector in the dataset
/// - Bit value 1: vector is included
/// - Bit value 0: vector is excluded
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Filter, Bitset};
/// use cuvs::{Resources, ManagedTensor};
/// use ndarray::Array1;
///
/// let res = Resources::new().unwrap();
/// let n_samples = 1000;
/// let bitset_size = (n_samples + 31) / 32;
///
/// // Create a bitset that filters out even-indexed vectors
/// let mut bitset_data = Array1::<u32>::zeros(bitset_size);
/// for i in 0..bitset_size {
///     bitset_data[i] = 0xAAAAAAAA; // binary: 10101010...
/// }
///
/// let bitset_tensor = ManagedTensor::from(&bitset_data).to_device(&res).unwrap();
/// let filter = Bitset::new(&bitset_tensor, n_samples);
/// ```
#[derive(Debug)]
pub struct Bitset<'a> {
    tensor: &'a ManagedTensor,
    _n_samples: usize,
}

impl<'a> Bitset<'a> {
    /// Create a new bitset filter
    ///
    /// # Arguments
    ///
    /// * `tensor` - Device tensor containing bitset data (uint32)
    /// * `n_samples` - Number of samples in the dataset being filtered
    ///
    /// The tensor should have shape `[(n_samples + 31) / 32]`
    pub fn new(tensor: &'a ManagedTensor, n_samples: usize) -> Self {
        Bitset {
            tensor,
            _n_samples: n_samples,
        }
    }
}

impl<'a> Filter for Bitset<'a> {
    fn into_ffi(&self) -> ffi::cuvsFilter {
        ffi::cuvsFilter {
            addr: self.tensor.as_ptr() as uintptr_t,
            type_: ffi::cuvsFilterType::BITSET,
        }
    }
}

/// Bitmap filter - applies different filters for each query
///
/// A bitmap allows per-query filtering in batch search operations.
/// Each query can have its own set of allowed/disallowed vectors.
///
/// # Layout
///
/// - Shape: `[n_queries * ((n_samples + 31) / 32)]` (uint32 elements)
/// - Each query has its own bitset of size `(n_samples + 31) / 32`
/// - Bit value 1: vector is included for this query
/// - Bit value 0: vector is excluded for this query
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Filter, Bitmap};
/// use cuvs::{Resources, ManagedTensor};
/// use ndarray::Array1;
///
/// let res = Resources::new().unwrap();
/// let n_samples = 1000;
/// let n_queries = 10;
/// let bits_per_query = (n_samples + 31) / 32;
/// let bitmap_size = n_queries * bits_per_query;
///
/// // Create a bitmap where each query filters out even-indexed vectors
/// let mut bitmap_data = Array1::<u32>::zeros(bitmap_size);
/// for i in 0..bitmap_size {
///     bitmap_data[i] = 0xAAAAAAAA; // binary: 10101010...
/// }
///
/// let bitmap_tensor = ManagedTensor::from(&bitmap_data).to_device(&res).unwrap();
/// let filter = Bitmap::new(&bitmap_tensor, n_queries, n_samples);
/// ```
#[derive(Debug)]
pub struct Bitmap<'a> {
    tensor: &'a ManagedTensor,
    _n_queries: usize,
    _n_samples: usize,
}

impl<'a> Bitmap<'a> {
    /// Create a new bitmap filter
    ///
    /// # Arguments
    ///
    /// * `tensor` - Device tensor containing bitmap data (uint32)
    /// * `n_queries` - Number of queries in the batch
    /// * `n_samples` - Number of samples in the dataset being filtered
    ///
    /// The tensor should have shape `[n_queries * ((n_samples + 31) / 32)]`
    pub fn new(tensor: &'a ManagedTensor, n_queries: usize, n_samples: usize) -> Self {
        Bitmap {
            tensor,
            _n_queries: n_queries,
            _n_samples: n_samples,
        }
    }
}

impl<'a> Filter for Bitmap<'a> {
    fn into_ffi(&self) -> ffi::cuvsFilter {
        ffi::cuvsFilter {
            addr: self.tensor.as_ptr() as uintptr_t,
            type_: ffi::cuvsFilterType::BITMAP,
        }
    }
}

// Re-export for convenience
use ffi::cuvs_sys as ffi;
type uintptr_t = usize;
