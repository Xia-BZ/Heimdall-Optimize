
#pragma once

#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>

struct not_my_pointer : public std::exception
{
    explicit not_my_pointer(void *p)
    {
        std::stringstream s;
        s << "Pointer" << p << " was not allocated by this allocator.";
        message = s.str();
    }
    virtual ~not_my_pointer() noexcept {}
    virtual const char *what() const noexcept override
    {
        return message.c_str();
    }

private:
    std::string message;
};

struct cached_allocator
{
    typedef char value_type;
    
    // 获取单例实例的静态方法
    static cached_allocator& get_instance()
    {
        static cached_allocator instance;
        return instance;
    }
    
    // 删除拷贝构造函数和赋值操作符
    cached_allocator(const cached_allocator&) = delete;
    cached_allocator& operator=(const cached_allocator&) = delete;
    
    ~cached_allocator()
    {
        free_all();
    }
    
    char *allocate(std::ptrdiff_t num_bytes)
    {
        // 智能内存对齐策略
        num_bytes = align_size(num_bytes);
        
        char* result = nullptr;
        
        // 使用双重检查锁定模式避免多线程竞争
        // 第一步：使用读锁快速检查是否有可用块
        {
            std::shared_lock<std::shared_mutex> read_lock(global_mutex_);
            auto it = free_blocks.find(num_bytes);
            if (it != free_blocks.end() && !it->second.empty()) {
                // 有可用块，需要升级到写锁来安全地取出
                read_lock.unlock();
                goto extract_block;
            }
            // 没有可用块，直接跳到分配新内存
        }
        
        // 分配新内存路径
        goto allocate_new_memory;
        
    extract_block:
        // 第二步：升级到写锁，进行双重检查并原子地取出块
        {
            std::unique_lock<std::shared_mutex> write_lock(global_mutex_);
            
            // 双重检查：在获得写锁后再次确认块仍然可用
            auto it = free_blocks.find(num_bytes);
            if (it != free_blocks.end() && !it->second.empty()) {
                // 原子地取出最后一个块
                result = it->second.back();
                it->second.pop_back();
                
                // 如果向量为空，移除这个条目以节省内存
                if (it->second.empty()) {
                    free_blocks.erase(it);
                }
                
                allocated_blocks[result] = num_bytes;
                
                return result;
            }
            // 如果在双重检查中发现块已被其他线程取走，继续分配新内存
        }
        
    allocate_new_memory:
        // 慢路径：分配新内存
        if (cudaMalloc((void **)&result, num_bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        
        // 记录新分配的块
        {
            std::unique_lock<std::shared_mutex> lock(global_mutex_);
            allocated_blocks[result] = num_bytes;
        }
        
        return result;
    }

    void deallocate(char *ptr, size_t)
    {
        if (ptr == nullptr) return;
        
        std::ptrdiff_t num_bytes = 0;
        
        // 查找并移除已分配块记录，然后放入缓存
        {
            std::unique_lock<std::shared_mutex> lock(global_mutex_);
            auto it = allocated_blocks.find(ptr);
            if (it == allocated_blocks.end()) {
                throw not_my_pointer(reinterpret_cast<void *>(ptr));
            }
            num_bytes = it->second;
            allocated_blocks.erase(it);
            
            // 直接放入缓存
            free_blocks[num_bytes].push_back(ptr);
        }
    }



private:
    // 私有化构造函数
    cached_allocator() {}
    
    // 全局缓存和已分配块的跟踪
    std::unordered_map<std::ptrdiff_t, std::vector<char *>> free_blocks;
    std::unordered_map<char *, std::ptrdiff_t> allocated_blocks;
    mutable std::shared_mutex global_mutex_; // 使用读写锁提升并发性能
    
    // 智能内存对齐策略
    static std::ptrdiff_t align_size(std::ptrdiff_t size) {
        // 根据内存大小选择不同的对齐策略
        std::ptrdiff_t alignment;
        if (size <= 1024) {
            // 小内存块：64字节对齐（CPU缓存行大小）
            alignment = 64;
        } else if (size <= 16384) {
            // 中等内存块：256字节对齐（GPU内存合并访问优化）
            alignment = 256;
        } else {
            // 大内存块：1024字节对齐（大块管理优化）
            alignment = 1024;
        }
        return ((size + alignment - 1) / alignment) * alignment;
    }
    
    void free_all()
    {
        std::unique_lock<std::shared_mutex> lock(global_mutex_);
        
        // 清理缓存中的块
        for (auto &pair : free_blocks)
        {
            for (auto ptr : pair.second)
            {
                cudaFree(ptr);
            }
        }
        free_blocks.clear();
        
        // 清理已分配但未释放的块
        for (auto &pair : allocated_blocks)
        {
            cudaFree(pair.first);
        }
        allocated_blocks.clear();
    }
};