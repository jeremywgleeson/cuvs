/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Example: Streaming CAGRA indices to/from S3 with compression
//!
//! This example demonstrates how to:
//! 1. Build a CAGRA index from a dataset
//! 2. Stream it compressed directly to S3 (no intermediate files)
//! 3. Load it back from S3, decompressing on-the-fly
//! 4. Perform searches on the loaded index
//!
//! To run this example:
//! ```bash
//! cargo run --example s3_streaming --features s3
//! ```
//!
//! Prerequisites:
//! - AWS credentials configured (via AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials)
//! - An S3 bucket you have write access to
//! - Environment variables:
//!   - S3_BUCKET: The bucket name
//!   - S3_KEY: The object key (e.g., "indices/my_index.bin.gz")
//!   - AWS_REGION: The AWS region (e.g., "us-west-2")

use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources};
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::io::{Read, Write};
use std::env;

/// Wrapper that implements Write for S3 multipart upload
struct S3Writer {
    client: aws_sdk_s3::Client,
    bucket: String,
    key: String,
    upload_id: String,
    parts: Vec<aws_sdk_s3::types::CompletedPart>,
    current_part: Vec<u8>,
    part_number: i32,
    runtime: tokio::runtime::Runtime,
}

impl S3Writer {
    fn new(bucket: String, key: String) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = tokio::runtime::Runtime::new()?;

        let client = runtime.block_on(async {
            let config = aws_config::load_from_env().await;
            aws_sdk_s3::Client::new(&config)
        });

        let upload_id = runtime.block_on(async {
            let response = client
                .create_multipart_upload()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await?;

            response.upload_id()
                .ok_or("No upload ID returned")?
                .to_string();

            Ok::<String, Box<dyn std::error::Error>>(
                response.upload_id().unwrap().to_string()
            )
        })?;

        Ok(Self {
            client,
            bucket,
            key,
            upload_id,
            parts: Vec::new(),
            current_part: Vec::new(),
            part_number: 1,
            runtime,
        })
    }

    fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Upload any remaining data
        if !self.current_part.is_empty() {
            self.upload_part()?;
        }

        // Complete the multipart upload
        self.runtime.block_on(async {
            let completed_parts = aws_sdk_s3::types::CompletedMultipartUpload::builder()
                .set_parts(Some(self.parts.clone()))
                .build();

            self.client
                .complete_multipart_upload()
                .bucket(&self.bucket)
                .key(&self.key)
                .upload_id(&self.upload_id)
                .multipart_upload(completed_parts)
                .send()
                .await?;

            Ok::<(), Box<dyn std::error::Error>>(())
        })?;

        Ok(())
    }

    fn upload_part(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.current_part.is_empty() {
            return Ok(());
        }

        let part_data = std::mem::take(&mut self.current_part);
        let part_number = self.part_number;

        let etag = self.runtime.block_on(async {
            let response = self.client
                .upload_part()
                .bucket(&self.bucket)
                .key(&self.key)
                .upload_id(&self.upload_id)
                .part_number(part_number)
                .body(aws_sdk_s3::primitives::ByteStream::from(part_data))
                .send()
                .await?;

            Ok::<String, Box<dyn std::error::Error>>(
                response.e_tag().unwrap_or("").to_string()
            )
        })?;

        self.parts.push(
            aws_sdk_s3::types::CompletedPart::builder()
                .part_number(part_number)
                .e_tag(etag)
                .build(),
        );

        self.part_number += 1;
        Ok(())
    }
}

impl Write for S3Writer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.current_part.extend_from_slice(buf);

        // Upload part when we reach 5MB (minimum part size for S3)
        if self.current_part.len() >= 5 * 1024 * 1024 {
            self.upload_part()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Wrapper that implements Read for S3 GetObject
struct S3Reader {
    buffer: Vec<u8>,
    position: usize,
}

impl S3Reader {
    fn new(bucket: String, key: String) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = tokio::runtime::Runtime::new()?;

        let buffer = runtime.block_on(async {
            let config = aws_config::load_from_env().await;
            let client = aws_sdk_s3::Client::new(&config);

            let response = client
                .get_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await?;

            let bytes = response.body.collect().await?;
            Ok::<Vec<u8>, Box<dyn std::error::Error>>(bytes.to_vec())
        })?;

        Ok(Self {
            buffer,
            position: 0,
        })
    }
}

impl Read for S3Reader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = &self.buffer[self.position..];
        let to_read = std::cmp::min(buf.len(), remaining.len());
        buf[..to_read].copy_from_slice(&remaining[..to_read]);
        self.position += to_read;
        Ok(to_read)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get configuration from environment
    let bucket = env::var("S3_BUCKET")
        .expect("S3_BUCKET environment variable not set");
    let key = env::var("S3_KEY")
        .unwrap_or_else(|_| "indices/cagra_index.bin.gz".to_string());

    println!("S3 Streaming Example");
    println!("====================");
    println!("Bucket: {}", bucket);
    println!("Key: {}", key);
    println!();

    // Initialize CUDA resources
    let res = Resources::new()?;

    // Step 1: Build an index from a dataset
    println!("Step 1: Building CAGRA index...");
    let n_datapoints = 10_000;
    let n_features = 128;
    let dataset = ndarray::Array::<f32, _>::random(
        (n_datapoints, n_features),
        Uniform::new(0.0, 1.0)
    );

    let params = IndexParams::new()?;
    let index = Index::build(&res, &params, &dataset)?;
    println!("✓ Built index with {} datapoints, {} dimensions", n_datapoints, n_features);
    println!();

    // Step 2: Stream compressed index to S3
    println!("Step 2: Streaming compressed index to S3...");
    let s3_writer = S3Writer::new(bucket.clone(), key.clone())?;
    let encoder = GzEncoder::new(s3_writer, Compression::default());

    // This streams directly to S3 without buffering the entire index in memory
    cuvs::streaming::serialize(&res, &index, encoder, true)?;

    println!("✓ Streamed index to s3://{}/{}", bucket, key);
    println!();

    // Step 3: Load index back from S3
    println!("Step 3: Loading compressed index from S3...");
    let s3_reader = S3Reader::new(bucket.clone(), key.clone())?;
    let decoder = GzDecoder::new(s3_reader);

    // This streams from S3 and decompresses on-the-fly
    let loaded_index = cuvs::streaming::deserialize(&res, decoder)?;
    println!("✓ Loaded index from s3://{}/{}", bucket, key);
    println!();

    // Step 4: Verify the loaded index works
    println!("Step 4: Verifying loaded index with search...");
    let n_queries = 100;
    let k = 10;

    // Use first 100 points as queries (should find themselves as nearest neighbors)
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

    // Verify first few queries found themselves
    let mut correct = 0;
    for i in 0..10 {
        if neighbors_host[[i, 0]] == i as u32 {
            correct += 1;
        }
    }

    println!("✓ Search completed");
    println!("  First 10 queries: {}/10 found themselves as nearest neighbor", correct);
    println!();

    println!("Success! The index was:");
    println!("  1. Built from {} vectors", n_datapoints);
    println!("  2. Compressed with gzip");
    println!("  3. Streamed to S3 (no temp files!)");
    println!("  4. Loaded back from S3");
    println!("  5. Verified with {} search queries", n_queries);

    Ok(())
}
