/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for index serialization/deserialization

use cuvs::cagra::{Index as CagraIndex, IndexParams as CagraIndexParams, SearchParams};
use cuvs::ivf_flat::{Index as IvfFlatIndex, IndexParams as IvfFlatIndexParams};
use cuvs::ivf_pq::{Index as IvfPqIndex, IndexParams as IvfPqIndexParams};
use cuvs::{ManagedTensor, Resources};
use ndarray::s;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::fs;
use tempfile::NamedTempFile;

/// Test CAGRA file-based serialization and deserialization
#[test]
fn test_cagra_file_serialize_deserialize() {
    let res = Resources::new().unwrap();
    let params = CagraIndexParams::new().unwrap();

    // Build index
    let n_datapoints = 256;
    let n_features = 16;
    let dataset = ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
    let index = CagraIndex::build(&res, &params, &dataset).expect("Failed to build CAGRA index");

    // Serialize to temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path();
    index.serialize(&res, temp_path, true).expect("Failed to serialize index");

    // Verify file exists and has content
    let metadata = fs::metadata(temp_path).unwrap();
    assert!(metadata.len() > 0, "Serialized file should not be empty");

    // Deserialize
    let loaded_index = CagraIndex::deserialize(&res, temp_path).expect("Failed to deserialize index");

    // Verify loaded index works by searching
    let n_queries = 4;
    let k = 10;
    let queries = dataset.slice(s![0..n_queries, ..]);

    let queries_device = ManagedTensor::from(&queries).to_device(&res).unwrap();
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

    let search_params = SearchParams::new().unwrap();
    loaded_index
        .search(&res, &search_params, &queries_device, &neighbors, &distances)
        .expect("Search failed");

    neighbors.to_host(&res, &mut neighbors_host).unwrap();

    // Verify results - queries should find themselves as nearest neighbors
    assert_eq!(neighbors_host[[0, 0]], 0);
    assert_eq!(neighbors_host[[1, 0]], 1);
    assert_eq!(neighbors_host[[2, 0]], 2);
    assert_eq!(neighbors_host[[3, 0]], 3);
}

/// Test CAGRA serialization with and without dataset
#[test]
fn test_cagra_serialize_with_without_dataset() {
    let res = Resources::new().unwrap();
    let params = CagraIndexParams::new().unwrap();

    let dataset = ndarray::Array::<f32, _>::random((256, 16), Uniform::new(0., 1.0));
    let index = CagraIndex::build(&res, &params, &dataset).unwrap();

    // Serialize with dataset
    let temp_with = NamedTempFile::new().unwrap();
    index.serialize(&res, temp_with.path(), true).unwrap();
    let size_with = fs::metadata(temp_with.path()).unwrap().len();

    // Serialize without dataset
    let temp_without = NamedTempFile::new().unwrap();
    index.serialize(&res, temp_without.path(), false).unwrap();
    let size_without = fs::metadata(temp_without.path()).unwrap().len();

    // Without dataset should be smaller
    assert!(size_without < size_with,
           "Index without dataset ({} bytes) should be smaller than with dataset ({} bytes)",
           size_without, size_with);
}

/// Test IVF-Flat file-based serialization and deserialization
#[test]
fn test_ivf_flat_file_serialize_deserialize() {
    let res = Resources::new().unwrap();
    let params = IvfFlatIndexParams::new().unwrap().set_n_lists(64);

    // Build index
    let n_datapoints = 1024;
    let n_features = 16;
    let dataset = ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
    let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();
    let index = IvfFlatIndex::build(&res, &params, dataset_device).expect("Failed to build IVF-Flat index");

    // Serialize
    let temp_file = NamedTempFile::new().unwrap();
    index.serialize(&res, temp_file.path()).expect("Failed to serialize index");

    // Verify file exists
    assert!(fs::metadata(temp_file.path()).unwrap().len() > 0);

    // Deserialize
    let loaded_index = IvfFlatIndex::deserialize(&res, temp_file.path())
        .expect("Failed to deserialize index");

    // Test search
    let n_queries = 4;
    let k = 10;
    let queries = dataset.slice(s![0..n_queries, ..]);
    let queries_device = ManagedTensor::from(&queries).to_device(&res).unwrap();

    let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

    let search_params = cuvs::ivf_flat::SearchParams::new().unwrap();
    loaded_index
        .search(&res, &search_params, &queries_device, &neighbors, &distances)
        .expect("Search failed");

    neighbors.to_host(&res, &mut neighbors_host).unwrap();

    // Verify first few results
    assert_eq!(neighbors_host[[0, 0]], 0);
    assert_eq!(neighbors_host[[1, 0]], 1);
}

/// Test IVF-PQ file-based serialization and deserialization
#[test]
fn test_ivf_pq_file_serialize_deserialize() {
    let res = Resources::new().unwrap();
    let params = IvfPqIndexParams::new().unwrap().set_n_lists(64);

    // Build index
    let n_datapoints = 1024;
    let n_features = 16;
    let dataset = ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
    let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();
    let index = IvfPqIndex::build(&res, &params, dataset_device).expect("Failed to build IVF-PQ index");

    // Serialize
    let temp_file = NamedTempFile::new().unwrap();
    index.serialize(&res, temp_file.path()).expect("Failed to serialize index");

    // Verify file exists
    assert!(fs::metadata(temp_file.path()).unwrap().len() > 0);

    // Deserialize
    let loaded_index = IvfPqIndex::deserialize(&res, temp_file.path())
        .expect("Failed to deserialize index");

    // Test search
    let n_queries = 4;
    let k = 10;
    let queries = dataset.slice(s![0..n_queries, ..]);
    let queries_device = ManagedTensor::from(&queries).to_device(&res).unwrap();

    let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

    let search_params = cuvs::ivf_pq::SearchParams::new().unwrap();
    loaded_index
        .search(&res, &search_params, &queries_device, &neighbors, &distances, None)
        .expect("Search failed");

    neighbors.to_host(&res, &mut neighbors_host).unwrap();

    // Verify first few results
    assert_eq!(neighbors_host[[0, 0]], 0);
    assert_eq!(neighbors_host[[1, 0]], 1);
}

/// Test that serialization survives across multiple serialize/deserialize cycles
#[test]
fn test_cagra_multiple_serialize_cycles() {
    let res = Resources::new().unwrap();
    let params = CagraIndexParams::new().unwrap();

    let dataset = ndarray::Array::<f32, _>::random((256, 16), Uniform::new(0., 1.0));
    let index1 = CagraIndex::build(&res, &params, &dataset).unwrap();

    // First cycle
    let temp1 = NamedTempFile::new().unwrap();
    index1.serialize(&res, temp1.path(), true).unwrap();
    let index2 = CagraIndex::deserialize(&res, temp1.path()).unwrap();

    // Second cycle
    let temp2 = NamedTempFile::new().unwrap();
    index2.serialize(&res, temp2.path(), true).unwrap();
    let index3 = CagraIndex::deserialize(&res, temp2.path()).unwrap();

    // Third cycle
    let temp3 = NamedTempFile::new().unwrap();
    index3.serialize(&res, temp3.path(), true).unwrap();
    let index4 = CagraIndex::deserialize(&res, temp3.path()).unwrap();

    // All serialized files should be the same size
    let size1 = fs::metadata(temp1.path()).unwrap().len();
    let size2 = fs::metadata(temp2.path()).unwrap().len();
    let size3 = fs::metadata(temp3.path()).unwrap().len();

    assert_eq!(size1, size2, "Serialized sizes should match across cycles");
    assert_eq!(size2, size3, "Serialized sizes should match across cycles");

    // Verify final index works
    let queries = dataset.slice(s![0..4, ..]);
    let queries_device = ManagedTensor::from(&queries).to_device(&res).unwrap();
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((4, 10));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
    let mut distances_host = ndarray::Array::<f32, _>::zeros((4, 10));
    let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

    let search_params = SearchParams::new().unwrap();
    index4
        .search(&res, &search_params, &queries_device, &neighbors, &distances)
        .unwrap();

    neighbors.to_host(&res, &mut neighbors_host).unwrap();
    assert_eq!(neighbors_host[[0, 0]], 0);
}
