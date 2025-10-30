/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{stderr, Write};

use crate::cagra::{IndexParams, SearchParams};
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// CAGRA ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsCagraIndex_t);

impl Index {
    /// Builds a new Index from the dataset for efficient search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build<T: Into<ManagedTensor>>(
        res: &Resources,
        params: &IndexParams,
        dataset: T,
    ) -> Result<Index> {
        let dataset: ManagedTensor = dataset.into();
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                index.0,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsCagraIndex_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Perform a Approximate Nearest Neighbors search on the Index
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters to use in searching the index
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    pub fn search(
        self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            let prefilter = ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
            };

            check_cuvs(ffi::cuvsCagraSearch(
                res.0,
                params.0,
                self.0,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                prefilter,
            ))
        }
    }

    /// Serialize the CAGRA index to a file
    ///
    /// Saves the index to disk for later reuse. This is useful for:
    /// - Persisting expensive-to-build indices
    /// - Sharing indices between processes or machines
    /// - Reducing startup time by loading pre-built indices
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - Path to the file where the index will be saved
    /// * `include_dataset` - Whether to include the dataset vectors in the serialized output.
    ///   Set to `true` if you need the original vectors, `false` to save space.
    ///
    /// # Performance
    ///
    /// Serialization writes incrementally to disk, so memory usage is bounded.
    /// For a 10M vector index with 128 dimensions:
    /// - With dataset: ~5-10 GB file size
    /// - Without dataset: ~100-500 MB file size (only graph structure)
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```no_run
    /// use cuvs::cagra::{Index, IndexParams};
    /// use cuvs::resources::Resources;
    /// # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
    ///
    /// let res = Resources::new().unwrap();
    /// let params = IndexParams::new().unwrap();
    /// let index = Index::build(&res, &params, &dataset).unwrap();
    ///
    /// // Save with dataset for complete index
    /// index.serialize(&res, "my_index.bin", true).unwrap();
    /// ```
    ///
    /// ## Save without dataset to reduce file size
    ///
    /// ```no_run
    /// # use cuvs::cagra::{Index, IndexParams};
    /// # use cuvs::resources::Resources;
    /// # let res = Resources::new().unwrap();
    /// # let params = IndexParams::new().unwrap();
    /// # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
    /// # let index = Index::build(&res, &params, &dataset).unwrap();
    ///
    /// // Save only graph structure (smaller file)
    /// index.serialize(&res, "graph_only.bin", false).unwrap();
    /// ```
    ///
    /// ## For streaming serialization, see [`crate::streaming`]
    ///
    /// ```no_run
    /// use cuvs::streaming;
    /// # use cuvs::cagra::{Index, IndexParams};
    /// # use cuvs::resources::Resources;
    /// # let res = Resources::new().unwrap();
    /// # let params = IndexParams::new().unwrap();
    /// # let dataset = ndarray::Array::<f32, _>::zeros((100, 10));
    /// # let index = Index::build(&res, &params, &dataset).unwrap();
    /// use std::fs::File;
    ///
    /// // Stream to compressed file
    /// use flate2::write::GzEncoder;
    /// use flate2::Compression;
    ///
    /// let file = File::create("index.bin.gz").unwrap();
    /// let encoder = GzEncoder::new(file, Compression::default());
    /// streaming::serialize_to_writer(&res, &index, encoder, true).unwrap();
    /// ```
    pub fn serialize(
        &self,
        res: &Resources,
        filename: impl AsRef<std::path::Path>,
        include_dataset: bool,
    ) -> Result<()> {
        use std::ffi::CString;
        let filename = CString::new(
            filename
                .as_ref()
                .to_str()
                .ok_or_else(|| crate::error::Error::new("Invalid filename"))?,
        )
        .map_err(|_| crate::error::Error::new("Invalid filename with null bytes"))?;

        unsafe {
            check_cuvs(ffi::cuvsCagraSerialize(
                res.0,
                filename.as_ptr(),
                self.0,
                include_dataset,
            ))
        }
    }

    /// Deserialize a CAGRA index from a file
    ///
    /// Loads a previously saved index from disk. The loaded index is immediately
    /// ready for searching without any additional build steps.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use (must match the GPU where index will be used)
    /// * `filename` - Path to the file containing the serialized index
    ///
    /// # Returns
    ///
    /// A fully constructed index ready for searching
    ///
    /// # Performance
    ///
    /// Loading is I/O bound. For large indices (>1GB), consider:
    /// - Using fast storage (NVMe SSD)
    /// - Streaming from compressed files (see [`crate::streaming`])
    /// - Pre-loading indices at application startup
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```no_run
    /// use cuvs::cagra::Index;
    /// use cuvs::resources::Resources;
    ///
    /// let res = Resources::new().unwrap();
    /// let index = Index::deserialize(&res, "my_index.bin").unwrap();
    ///
    /// // Index is ready to use immediately
    /// // ... perform searches ...
    /// ```
    ///
    /// ## Load and search
    ///
    /// ```no_run
    /// use cuvs::cagra::{Index, SearchParams};
    /// use cuvs::{Resources, ManagedTensor};
    /// # let queries = ndarray::Array::<f32, _>::zeros((10, 128));
    ///
    /// let res = Resources::new().unwrap();
    /// let index = Index::deserialize(&res, "index.bin").unwrap();
    ///
    /// // Perform search
    /// let search_params = SearchParams::new().unwrap();
    /// let queries_device = ManagedTensor::from(&queries).to_device(&res).unwrap();
    /// let mut neighbors = ndarray::Array::<u32, _>::zeros((10, 10));
    /// let neighbors_device = ManagedTensor::from(&neighbors).to_device(&res).unwrap();
    /// let mut distances = ndarray::Array::<f32, _>::zeros((10, 10));
    /// let distances_device = ManagedTensor::from(&distances).to_device(&res).unwrap();
    ///
    /// index.search(&res, &search_params, &queries_device, &neighbors_device, &distances_device).unwrap();
    /// ```
    ///
    /// ## For streaming deserialization, see [`crate::streaming`]
    ///
    /// ```no_run
    /// use cuvs::streaming;
    /// use cuvs::resources::Resources;
    /// use std::fs::File;
    /// use flate2::read::GzDecoder;
    ///
    /// let res = Resources::new().unwrap();
    ///
    /// // Load from compressed file
    /// let file = File::open("index.bin.gz").unwrap();
    /// let decoder = GzDecoder::new(file);
    /// let index = streaming::deserialize_from_reader(&res, decoder).unwrap();
    /// ```
    pub fn deserialize(res: &Resources, filename: impl AsRef<std::path::Path>) -> Result<Index> {
        use std::ffi::CString;
        let filename = CString::new(
            filename
                .as_ref()
                .to_str()
                .ok_or_else(|| crate::error::Error::new("Invalid filename"))?,
        )
        .map_err(|_| crate::error::Error::new("Invalid filename with null bytes"))?;

        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserialize(
                res.0,
                filename.as_ptr(),
                index.0,
            ))?;
        }
        Ok(index)
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cagraIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // build the cagra index
        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);

        let k = 10;

        // CAGRA search API requires queries and outputs to be on device memory
        // copy query data over, and allocate new device memory for the distances/ neighbors
        // outputs
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        let search_params = SearchParams::new().unwrap();

        index
            .search(&res, &search_params, &queries, &neighbors, &distances)
            .unwrap();

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // nearest neighbors should be themselves, since queries are from the
        // dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::new().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_compression() {
        use crate::cagra::CompressionParams;
        let build_params = IndexParams::new()
            .unwrap()
            .set_compression(CompressionParams::new().unwrap());
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_serialize() {
        let build_params = IndexParams::new().unwrap();
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // build the cagra index
        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // Serialize the index
        let temp_file = std::env::temp_dir().join("test_cagra_index.bin");
        index
            .serialize(&res, &temp_file, true)
            .expect("failed to serialize index");

        // Deserialize the index
        let loaded_index =
            Index::deserialize(&res, &temp_file).expect("failed to deserialize index");

        // Test that the loaded index works
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let k = 10;

        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        let search_params = SearchParams::new().unwrap();

        loaded_index
            .search(&res, &search_params, &queries, &neighbors, &distances)
            .unwrap();

        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // Verify results
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }
}
