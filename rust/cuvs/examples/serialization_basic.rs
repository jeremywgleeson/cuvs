/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Example: Basic file-based serialization
//!
//! This example demonstrates the simple file-based API:
//! 1. Build a CAGRA index
//! 2. Serialize to a file
//! 3. Deserialize from the file
//! 4. Verify it works
//!
//! For streaming to compression/S3/etc, see the streaming_basic.rs and s3_streaming.rs examples.
//!
//! To run:
//! ```bash
//! cargo run --example serialization_basic
//! ```

use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CAGRA File Serialization Example");
    println!("=================================\n");

    // Initialize CUDA resources
    let res = Resources::new()?;

    // Build an index
    println!("Building index...");
    let n_datapoints = 10_000;
    let n_features = 128;
    let dataset = ndarray::Array::<f32, _>::random(
        (n_datapoints, n_features),
        Uniform::new(0.0, 1.0)
    );

    let params = IndexParams::new()?;
    let index = Index::build(&res, &params, &dataset)?;
    println!("✓ Built index with {} vectors of {} dimensions\n", n_datapoints, n_features);

    // Serialize to file - simple API
    println!("Serializing to file...");
    let output_path = "cagra_index.bin";

    // This is the simple file-based API
    index.serialize(&res, output_path, true)?;

    let file_size = std::fs::metadata(output_path)?.len();
    println!("✓ Serialized to {} ({:.2} MB)\n", output_path, file_size as f64 / 1_000_000.0);

    // Deserialize from file - simple API
    println!("Deserializing from file...");

    // This is the simple file-based API
    let loaded_index = Index::deserialize(&res, output_path)?;
    println!("✓ Deserialized from {}\n", output_path);

    // Verify with a search
    println!("Verifying with search...");
    let n_queries = 10;
    let k = 10;
    let queries = dataset.slice(ndarray::s![0..n_queries, ..]);

    let queries_device = ManagedTensor::from(&queries).to_device(&res)?;
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;
    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res)?;

    let search_params = SearchParams::new()?;
    loaded_index.search(&res, &search_params, &queries_device, &neighbors, &distances)?;

    neighbors.to_host(&res, &mut neighbors_host)?;
    distances.to_host(&res, &mut distances_host)?;

    // Verify queries found themselves as nearest neighbors
    println!("✓ Search completed");
    println!("\nResults (first 5 queries):");
    for i in 0..5 {
        let found_self = neighbors_host[[i, 0]] == i as u32;
        let distance = distances_host[[i, 0]];
        println!("  Query {}: nearest neighbor = {}, distance = {:.6} {}",
                 i,
                 neighbors_host[[i, 0]],
                 distance,
                 if found_self { "✓" } else { "✗" });
    }

    // Also demonstrate serializing without the dataset (smaller file)
    println!("\nSerializing without dataset...");
    let graph_only_path = "cagra_graph_only.bin";
    index.serialize(&res, graph_only_path, false)?;

    let graph_only_size = std::fs::metadata(graph_only_path)?.len();
    println!("✓ Graph-only file: {} ({:.2} MB)", graph_only_path, graph_only_size as f64 / 1_000_000.0);
    println!("  Savings: {:.1}% smaller", (1.0 - graph_only_size as f64 / file_size as f64) * 100.0);

    // Cleanup
    std::fs::remove_file(output_path)?;
    std::fs::remove_file(graph_only_path)?;

    println!("\nSuccess!");
    println!("\nNote: For streaming to compression/S3/etc, see:");
    println!("  - cargo run --example streaming_basic");
    println!("  - cargo run --example s3_streaming");

    Ok(())
}
