/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Example: Basic streaming with compression
//!
//! This example demonstrates the core streaming API:
//! 1. Build a CAGRA index
//! 2. Stream it to a compressed file
//! 3. Load it back from the compressed file
//! 4. Verify it works
//!
//! To run:
//! ```bash
//! cargo run --example streaming_basic
//! ```

use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources};
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::fs::File;
use std::io::BufWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CAGRA Streaming Serialization Example");
    println!("======================================\n");

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

    // Serialize to compressed file using streaming API
    println!("Serializing to compressed file...");
    let output_path = "index.bin.gz";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    let encoder = GzEncoder::new(writer, Compression::default());

    // This is the key API: streaming::serialize()
    // It accepts any Write implementation, no buffering in memory
    cuvs::streaming::serialize(&res, &index, encoder, true)?;

    let file_size = std::fs::metadata(output_path)?.len();
    println!("✓ Serialized to {} ({:.2} MB)\n", output_path, file_size as f64 / 1_000_000.0);

    // Deserialize from compressed file
    println!("Deserializing from compressed file...");
    let file = File::open(output_path)?;
    let decoder = GzDecoder::new(file);

    // This is the key API: streaming::deserialize()
    // It accepts any Read implementation, no buffering in memory
    let loaded_index = cuvs::streaming::deserialize(&res, decoder)?;
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

    // Cleanup
    std::fs::remove_file(output_path)?;

    println!("\nSuccess!");
    Ok(())
}
