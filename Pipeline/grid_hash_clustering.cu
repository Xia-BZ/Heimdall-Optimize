/***************************************************************************
 * GPU优化聚类算法 - 针对脉冲星候选体聚类优化
 * 使用共享内存优化的O(n²)聚类算法
 ***************************************************************************/

#include "hd/types.h"
#include "hd/are_coincident.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/count.h>

// 定义错误代码
typedef int hd_error;
#define HD_NO_ERROR 0
#define HD_UNKNOWN_ERROR -1

// GPU优化的聚类核函数 - 使用共享内存提升GPU执行效率
__global__ void __launch_bounds__(256, 2) optimizedClusteringKernel(
    int num_candidates,
    const hd_size* __restrict__ samp_inds, 
    const hd_size* __restrict__ begins, 
    const hd_size* __restrict__ ends,
    const hd_size* __restrict__ filter_inds, 
    const hd_size* __restrict__ dm_inds,
    hd_size* __restrict__ labels,
    hd_size time_tol, hd_size filter_tol, hd_size dm_tol) {
    
    // 使用共享内存缓存候选体数据，减少全局内存访问
    const int SHARED_SIZE = 256;
    __shared__ hd_size shared_samp_inds[SHARED_SIZE];
    __shared__ hd_size shared_begins[SHARED_SIZE];
    __shared__ hd_size shared_ends[SHARED_SIZE];
    __shared__ hd_size shared_filter_inds[SHARED_SIZE];
    __shared__ hd_size shared_dm_inds[SHARED_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx >= num_candidates) return;
    
    // 当前线程处理的候选体数据（缓存到寄存器）
    const hd_size samp_i = samp_inds[idx];
    const hd_size begin_i = begins[idx];
    const hd_size end_i = ends[idx];
    const hd_size filter_i = filter_inds[idx]; 
    const hd_size dm_i = dm_inds[idx];
    
    // 分块处理所有候选体，每次处理SHARED_SIZE个
    for (int block_start = 0; block_start < num_candidates; block_start += SHARED_SIZE) {
        int block_end = min(block_start + SHARED_SIZE, num_candidates);
        int block_size = block_end - block_start;
        
        // 协作加载数据到共享内存
        for (int offset = tid; offset < block_size; offset += blockDim.x) {
            int global_idx = block_start + offset;
            if (global_idx < num_candidates) {
                shared_samp_inds[offset] = samp_inds[global_idx];
                shared_begins[offset] = begins[global_idx];
                shared_ends[offset] = ends[global_idx];
                shared_filter_inds[offset] = filter_inds[global_idx];
                shared_dm_inds[offset] = dm_inds[global_idx];
            }
        }
        
        __syncthreads();  // 确保共享内存加载完成
        
        // 在共享内存中进行邻居搜索
        for (int j = 0; j < block_size; ++j) {
            int global_j = block_start + j;
            if (global_j == idx) continue;  // 跳过自己
            
            // 从共享内存读取邻居数据
            const hd_size samp_j = shared_samp_inds[j];
            const hd_size begin_j = shared_begins[j];
            const hd_size end_j = shared_ends[j];
            const hd_size filter_j = shared_filter_inds[j];
            const hd_size dm_j = shared_dm_inds[j];
            
            // 检查是否为邻居
            if (are_coincident(samp_i, samp_j,
                              begin_i, begin_j,
                              end_i, end_j,
                              filter_i, filter_j,
                              dm_i, dm_j,
                              time_tol, filter_tol, dm_tol)) {
                
                // 更新标签为较小值
                hd_size current_label_j = labels[global_j];
                
                #if defined(__LP64__) || defined(_WIN64)
                    atomicMin((unsigned long long*)&labels[idx], (unsigned long long)current_label_j);
                #else
                    atomicMin((unsigned int*)&labels[idx], (unsigned int)current_label_j);
                #endif
            }
        }
        
        __syncthreads();  // 同步确保本块处理完成再进入下一块
    }
}

// 等价链追踪核函数（路径压缩）
__global__ void traceEquivalencyChainKernel(hd_size* labels, int num_candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    // 缓存当前标签到寄存器，减少内存访问
    hd_size cur_label = labels[idx];
    hd_size original_label = cur_label;
    
    // 路径压缩 - 追踪到根标签
    while (labels[cur_label] != cur_label) {
        hd_size next_label = labels[cur_label];
        cur_label = next_label;
    }
    
    // 只在标签发生变化时才进行原子更新
    if (cur_label != original_label) {
        #if defined(__LP64__) || defined(_WIN64)
            atomicExch((unsigned long long*)&labels[idx], (unsigned long long)cur_label);
        #else
            atomicExch((unsigned int*)&labels[idx], (unsigned int)cur_label);
        #endif
    }
}

// 主聚类函数 - GPU优化版本
hd_error label_candidate_clusters_optimized(hd_size            count,
                                           ConstRawCandidates d_cands,
                                           hd_size            time_tol,
                                           hd_size            filter_tol,
                                           hd_size            dm_tol,
                                           hd_size*           d_labels,
                                           hd_size*           label_count) {
    
    if (count == 0) {
        *label_count = 0;
        return HD_NO_ERROR;
    }
    
    // 1. 初始化标签
    thrust::device_ptr<hd_size> d_labels_begin(d_labels);
    thrust::sequence(d_labels_begin, d_labels_begin + count);
    
    // 2. 执行聚类
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    optimizedClusteringKernel<<<grid, block>>>(
        count,
        d_cands.inds, d_cands.begins, d_cands.ends,
        d_cands.filter_inds, d_cands.dm_inds,
        d_labels, time_tol, filter_tol, dm_tol
    );
    
    cudaDeviceSynchronize();
    
    // 3. 等价链追踪
    traceEquivalencyChainKernel<<<grid, block>>>(d_labels, count);
    cudaDeviceSynchronize();
    
    // 4. 计算标签数量
    thrust::device_vector<int> d_label_roots(count);
    thrust::transform(d_labels_begin, d_labels_begin + count,
                      thrust::counting_iterator<hd_size>(0),
                      d_label_roots.begin(),
                      thrust::equal_to<hd_size>());
    *label_count = thrust::count_if(d_label_roots.begin(),
                                   d_label_roots.end(),
                                   thrust::identity<hd_size>());
    
    return HD_NO_ERROR;
} 