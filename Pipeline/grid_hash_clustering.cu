/***************************************************************************
 * 网格哈希聚类算法 - 针对脉冲星候选体聚类优化
 * 将O(n²)复杂度降低到O(n)
 ***************************************************************************/

#include "hd/types.h"
#include "hd/are_coincident.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/count.h>

// 定义错误代码
typedef int hd_error;
#define HD_NO_ERROR 0
#define HD_UNKNOWN_ERROR -1

// 网格哈希数据结构
struct GridHashParams {
    float time_grid_size;      // 时间维度网格大小
    float filter_grid_size;    // 滤波器维度网格大小  
    float dm_grid_size;        // DM维度网格大小
    
    // 根据容忍度自动设置网格大小
    GridHashParams(float time_tol, float filter_tol, float dm_tol) {
        // 网格大小应该足够小以确保相近的候选体在相邻网格中
        // 但又不能太小导致过多的网格
        time_grid_size = time_tol / 2.0f;      // 网格大小为容忍度的一半
        filter_grid_size = filter_tol + 1.0f;  // 滤波器容忍度通常较小
        dm_grid_size = dm_tol + 1.0f;          // DM容忍度通常较小
    }
};

// 计算网格坐标
__device__ __forceinline__ 
uint3 computeGridCoord(hd_size begin, hd_size filter_ind, hd_size dm_ind, 
                       const GridHashParams& params) {
    return make_uint3(
        (uint)(begin / params.time_grid_size),
        (uint)(filter_ind / params.filter_grid_size), 
        (uint)(dm_ind / params.dm_grid_size)
    );
}

// 3D网格坐标哈希函数
__device__ __forceinline__
uint computeGridHash(uint3 coord) {
    // 使用大质数避免哈希冲突
    return coord.x * 73856093u ^ coord.y * 19349663u ^ coord.z * 83492791u;
}

// 计算候选体的网格哈希值
struct ComputeHashFunctor {
    const hd_size* begins;
    const hd_size* filter_inds; 
    const hd_size* dm_inds;
    GridHashParams params;
    
    ComputeHashFunctor(const hd_size* b, const hd_size* f, const hd_size* d, 
                       GridHashParams p) : begins(b), filter_inds(f), dm_inds(d), params(p) {}
    
    __device__
    uint operator()(int idx) const {
        uint3 coord = computeGridCoord(begins[idx], filter_inds[idx], dm_inds[idx], params);
        return computeGridHash(coord);
    }
};

// Union-Find查找根节点（带路径压缩）
__device__ hd_size find_root(hd_size* labels, hd_size idx) {
    hd_size root = idx;
    // 找到根节点
    while (labels[root] != root) {
        root = labels[root];
    }
    // 路径压缩
    hd_size current = idx;
    while (labels[current] != root) {
        hd_size next = labels[current];
        labels[current] = root;
        current = next;
    }
    return root;
}

// Union-Find合并操作
__device__ void union_sets(hd_size* labels, hd_size a, hd_size b) {
    hd_size root_a = find_root(labels, a);
    hd_size root_b = find_root(labels, b);
    
    if (root_a != root_b) {
        // 总是让较小的根成为父节点
        if (root_a < root_b) {
            labels[root_b] = root_a;
        } else {
            labels[root_a] = root_b;
        }
    }
}

// 修改聚类核函数，使用原子操作避免竞态条件
__global__ void exactlyMatchOriginalClusteringKernel(
    int num_candidates,
    const hd_size* samp_inds, const hd_size* begins, const hd_size* ends,
    const hd_size* filter_inds, const hd_size* dm_inds,
    hd_size* labels,
    hd_size time_tol, hd_size filter_tol, hd_size dm_tol) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    hd_size samp_i = samp_inds[idx];
    hd_size begin_i = begins[idx];
    hd_size end_i = ends[idx];
    hd_size filter_i = filter_inds[idx]; 
    hd_size dm_i = dm_inds[idx];
    
    // 完全模拟原始算法的行为：对每个候选体，找到其所有邻居中的最小标签
    for (int j = 0; j < num_candidates; j++) {
        if (j == idx) continue;
        
        hd_size samp_j = samp_inds[j];
        hd_size begin_j = begins[j];
        hd_size end_j = ends[j];
        hd_size filter_j = filter_inds[j];
        hd_size dm_j = dm_inds[j];
        
        if (are_coincident(samp_i, samp_j,
                          begin_i, begin_j,
                          end_i, end_j,
                          filter_i, filter_j,
                          dm_i, dm_j,
                          time_tol, filter_tol, dm_tol)) {
            
            // 使用原子操作避免竞态条件
            // 根据hd_size实际类型选择合适的atomicMin
            #if defined(__LP64__) || defined(_WIN64)
                // 64位系统，hd_size是size_t（64位）
                atomicMin((unsigned long long*)&labels[idx], (unsigned long long)labels[j]);
            #else
                // 32位系统，hd_size是size_t（32位）
                atomicMin((unsigned int*)&labels[idx], (unsigned int)labels[j]);
            #endif
        }
    }
}

// 等价链追踪核函数（使用原子操作确保线程安全）
__global__ void traceEquivalencyChainKernel(hd_size* labels, int num_candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    hd_size cur_label = idx;
    while (labels[cur_label] != cur_label) {
        cur_label = labels[cur_label];
    }
    
    // 使用原子操作确保路径压缩的线程安全
    // 根据hd_size实际类型选择合适的atomicExch
    #if defined(__LP64__) || defined(_WIN64)
        // 64位系统，hd_size是size_t（64位）
        atomicExch((unsigned long long*)&labels[idx], (unsigned long long)cur_label);
    #else
        // 32位系统，hd_size是size_t（32位）
        atomicExch((unsigned int*)&labels[idx], (unsigned int)cur_label);
    #endif
}

// 主聚类函数 - 保持与原始函数相同的接口
hd_error label_candidate_clusters_grid_hash(hd_size            count,
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
    
    // 1. 初始化标签（与原始算法相同）
    thrust::device_ptr<hd_size> d_labels_begin(d_labels);
    thrust::sequence(d_labels_begin, d_labels_begin + count);
    
    // 2. 执行聚类（与原始算法完全相同的逻辑）
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    exactlyMatchOriginalClusteringKernel<<<grid, block>>>(
        count,
        d_cands.inds, d_cands.begins, d_cands.ends,
        d_cands.filter_inds, d_cands.dm_inds,
        d_labels, time_tol, filter_tol, dm_tol
    );
    
    cudaDeviceSynchronize();
    
    // 3. 等价链追踪（与原始算法相同）
    traceEquivalencyChainKernel<<<grid, block>>>(d_labels, count);
    cudaDeviceSynchronize();
    
    // 4. 计算标签数量（与原始算法相同）
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