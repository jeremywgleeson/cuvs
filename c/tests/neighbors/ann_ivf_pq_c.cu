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

#include <cuda.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include "neighbors/ann_utils.cuh"
#include <cuvs/neighbors/ivf_pq.h>

extern "C" void run_ivf_pq(int64_t n_rows,
                           int64_t n_queries,
                           int64_t n_dim,
                           uint32_t n_neighbors,
                           float* index_data,
                           float* query_data,
                           float* distances_data,
                           int64_t* neighbors_data,
                           cuvsDistanceType metric,
                           size_t n_probes,
                           size_t n_lists);

template <typename T>
void generate_random_data(T* devPtr, size_t size)
{
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
};

template <typename T, typename IdxT>
void recall_eval(T* query_data,
                 T* index_data,
                 IdxT* neighbors,
                 T* distances,
                 size_t n_queries,
                 size_t n_rows,
                 size_t n_dim,
                 size_t n_neighbors,
                 cuvsDistanceType metric,
                 size_t n_probes,
                 size_t n_lists)
{
  raft::handle_t handle;
  auto distances_ref = raft::make_device_matrix<T, IdxT>(handle, n_queries, n_neighbors);
  auto neighbors_ref = raft::make_device_matrix<IdxT, IdxT>(handle, n_queries, n_neighbors);
  cuvs::neighbors::naive_knn<T, T, IdxT>(
    handle,
    distances_ref.data_handle(),
    neighbors_ref.data_handle(),
    query_data,
    index_data,
    n_queries,
    n_rows,
    n_dim,
    n_neighbors,
    static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);
  std::vector<IdxT> neighbors_ref_h(size);
  std::vector<T> distances_ref_h(size);

  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_h.data(), neighbors, size, stream);
  raft::copy(distances_h.data(), distances, size, stream);
  raft::copy(neighbors_ref_h.data(), neighbors_ref.data_handle(), size, stream);
  raft::copy(distances_ref_h.data(), distances_ref.data_handle(), size, stream);

  // verify output
  double min_recall = static_cast<double>(n_probes) / static_cast<double>(n_lists);
  ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                               neighbors_h,
                                               distances_ref_h,
                                               distances_h,
                                               n_queries,
                                               n_neighbors,
                                               0.001,
                                               min_recall));
};

TEST(IvfPqC, BuildSearch)
{
  int64_t n_rows       = 8096;
  int64_t n_queries    = 128;
  int64_t n_dim        = 32;
  uint32_t n_neighbors = 8;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 20;
  size_t n_lists          = 1024;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  run_ivf_pq(n_rows,
             n_queries,
             n_dim,
             n_neighbors,
             index_data.data(),
             query_data.data(),
             distances_data.data(),
             neighbors_data.data(),
             metric,
             n_probes,
             n_lists);

  recall_eval(query_data.data(),
              index_data.data(),
              neighbors_data.data(),
              distances_data.data(),
              n_queries,
              n_rows,
              n_dim,
              n_neighbors,
              metric,
              n_probes,
              n_lists);
}

TEST(IvfPqC, BuildSearchBitsetFiltered)
{
  int64_t n_rows       = 1000;
  int64_t n_queries    = 10;
  int64_t n_dim        = 16;
  uint32_t n_neighbors = 10;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 50;
  size_t n_lists          = 100;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsIvfPqIndex_t index;
  cuvsIvfPqIndexCreate(&index);

  // build index
  cuvsIvfPqIndexParams_t build_params;
  cuvsIvfPqIndexParamsCreate(&build_params);
  build_params->metric  = metric;
  build_params->n_lists = n_lists;
  cuvsIvfPqBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = (void*)query_data.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = (void*)neighbors_data.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = (void*)distances_data.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // create bitset filter (0xAAAAAAAA filters out even indices)
  size_t bitset_size = (n_rows + 31) / 32;
  rmm::device_uvector<uint32_t> bitset_data(bitset_size, stream);
  std::vector<uint32_t> bitset_h(bitset_size, 0xAAAAAAAA);
  raft::copy(bitset_data.data(), bitset_h.data(), bitset_size, stream);

  DLManagedTensor bitset_tensor;
  bitset_tensor.dl_tensor.data               = (void*)bitset_data.data();
  bitset_tensor.dl_tensor.device.device_type = kDLCUDA;
  bitset_tensor.dl_tensor.ndim               = 1;
  bitset_tensor.dl_tensor.dtype.code         = kDLUInt;
  bitset_tensor.dl_tensor.dtype.bits         = 32;
  bitset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t bitset_shape[1]                    = {(int64_t)bitset_size};
  bitset_tensor.dl_tensor.shape              = bitset_shape;
  bitset_tensor.dl_tensor.strides            = NULL;

  // search index with BITSET filter
  cuvsIvfPqSearchParams_t search_params;
  cuvsIvfPqSearchParamsCreate(&search_params);
  search_params->n_probes = n_probes;
  cuvsFilter filter;
  filter.addr = (uintptr_t)&bitset_tensor;
  filter.type = BITSET;
  cuvsIvfPqSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

  // verify all neighbors are odd indices
  std::vector<int64_t> neighbors_h(n_queries * n_neighbors);
  raft::copy(neighbors_h.data(), neighbors_data.data(), n_queries * n_neighbors, stream);
  raft::resource::sync_stream(handle);

  for (size_t i = 0; i < n_queries * n_neighbors; i++) {
    ASSERT_TRUE(neighbors_h[i] % 2 == 1) << "Found even index " << neighbors_h[i];
  }

  // de-allocate index and res
  cuvsIvfPqSearchParamsDestroy(search_params);
  cuvsIvfPqIndexParamsDestroy(build_params);
  cuvsIvfPqIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(IvfPqC, BuildSearchBitmapFiltered)
{
  int64_t n_rows       = 1000;
  int64_t n_queries    = 10;
  int64_t n_dim        = 16;
  uint32_t n_neighbors = 10;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 50;
  size_t n_lists          = 100;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsIvfPqIndex_t index;
  cuvsIvfPqIndexCreate(&index);

  // build index
  cuvsIvfPqIndexParams_t build_params;
  cuvsIvfPqIndexParamsCreate(&build_params);
  build_params->metric  = metric;
  build_params->n_lists = n_lists;
  cuvsIvfPqBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = (void*)query_data.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = (void*)neighbors_data.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = (void*)distances_data.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // create bitmap filter (0xAAAAAAAA filters out even indices for each query)
  size_t bits_per_query = (n_rows + 31) / 32;
  size_t bitmap_size    = n_queries * bits_per_query;
  rmm::device_uvector<uint32_t> bitmap_data(bitmap_size, stream);
  std::vector<uint32_t> bitmap_h(bitmap_size, 0xAAAAAAAA);
  raft::copy(bitmap_data.data(), bitmap_h.data(), bitmap_size, stream);

  DLManagedTensor bitmap_tensor;
  bitmap_tensor.dl_tensor.data               = (void*)bitmap_data.data();
  bitmap_tensor.dl_tensor.device.device_type = kDLCUDA;
  bitmap_tensor.dl_tensor.ndim               = 1;
  bitmap_tensor.dl_tensor.dtype.code         = kDLUInt;
  bitmap_tensor.dl_tensor.dtype.bits         = 32;
  bitmap_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t bitmap_shape[1]                    = {(int64_t)bitmap_size};
  bitmap_tensor.dl_tensor.shape              = bitmap_shape;
  bitmap_tensor.dl_tensor.strides            = NULL;

  // search index with BITMAP filter
  cuvsIvfPqSearchParams_t search_params;
  cuvsIvfPqSearchParamsCreate(&search_params);
  search_params->n_probes = n_probes;
  cuvsFilter filter;
  filter.addr = (uintptr_t)&bitmap_tensor;
  filter.type = BITMAP;
  cuvsIvfPqSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

  // verify all neighbors are odd indices
  std::vector<int64_t> neighbors_h(n_queries * n_neighbors);
  raft::copy(neighbors_h.data(), neighbors_data.data(), n_queries * n_neighbors, stream);
  raft::resource::sync_stream(handle);

  for (size_t i = 0; i < n_queries * n_neighbors; i++) {
    ASSERT_TRUE(neighbors_h[i] % 2 == 1) << "Found even index " << neighbors_h[i];
  }

  // de-allocate index and res
  cuvsIvfPqSearchParamsDestroy(search_params);
  cuvsIvfPqIndexParamsDestroy(build_params);
  cuvsIvfPqIndexDestroy(index);
  cuvsResourcesDestroy(res);
}
