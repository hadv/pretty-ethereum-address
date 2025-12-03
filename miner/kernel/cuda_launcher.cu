/*
 * CUDA Library for CREATE2 Vanity Address Mining
 * This file provides all C-callable functions for the Go bindings
 *
 * Compile with: nvcc -c -o cuda_miner.o cuda_launcher.cu -arch=sm_50
 *               ar rcs libvaneth_cuda.a cuda_miner.o
 */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

// Include the kernel implementation
#include "keccak256.cu"

// Device info structure (must match Go side)
typedef struct {
    int index;
    char name[256];
    int compute_units;
    int max_threads_per_block;
    unsigned long long total_memory;
} CUDADeviceInfo;

// CUDA miner context
typedef struct {
    int device_index;
    unsigned char* d_data_template;
    unsigned char* d_pattern;
    unsigned char* d_result_salt;
    unsigned char* d_result_address;
    int* d_found;
    int batch_size;
    char device_name[256];
    // Cache for avoiding repeated copies
    unsigned char cached_template[85];
    unsigned char cached_pattern[20];
    int cached_pattern_length;
    int template_initialized;
    int pattern_initialized;
} CUDAMinerContext;

extern "C" {

// Get number of CUDA devices
// Returns device count on success, negative error code on failure
int get_cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // Return negative error code for debugging
        return -((int)err);
    }
    return count;
}

// Get device info
int get_cuda_device_info(int index, CUDADeviceInfo* info) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, index);
    if (err != cudaSuccess) {
        return -1;
    }
    info->index = index;
    strncpy(info->name, prop.name, 255);
    info->name[255] = '\0';
    info->compute_units = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->total_memory = (unsigned long long)prop.totalGlobalMem;
    return 0;
}

// Initialize CUDA miner
CUDAMinerContext* cuda_miner_init(int device_index, int batch_size) {
    cudaError_t err;

    // Set device
    err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
        return NULL;
    }

    CUDAMinerContext* ctx = (CUDAMinerContext*)malloc(sizeof(CUDAMinerContext));
    if (!ctx) return NULL;

    ctx->device_index = device_index;
    ctx->batch_size = batch_size;

    // Get device name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    strncpy(ctx->device_name, prop.name, 255);
    ctx->device_name[255] = '\0';

    // Allocate device memory
    err = cudaMalloc(&ctx->d_data_template, 85);
    if (err != cudaSuccess) { free(ctx); return NULL; }

    err = cudaMalloc(&ctx->d_pattern, 20);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_data_template);
        free(ctx); return NULL;
    }

    err = cudaMalloc(&ctx->d_result_salt, 12);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_data_template);
        cudaFree(ctx->d_pattern);
        free(ctx); return NULL;
    }

    err = cudaMalloc(&ctx->d_result_address, 20);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_data_template);
        cudaFree(ctx->d_pattern);
        cudaFree(ctx->d_result_salt);
        free(ctx); return NULL;
    }

    err = cudaMalloc(&ctx->d_found, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_data_template);
        cudaFree(ctx->d_pattern);
        cudaFree(ctx->d_result_salt);
        cudaFree(ctx->d_result_address);
        free(ctx); return NULL;
    }

    // Initialize cache flags
    ctx->template_initialized = 0;
    ctx->pattern_initialized = 0;
    ctx->cached_pattern_length = 0;
    memset(ctx->cached_template, 0, 85);
    memset(ctx->cached_pattern, 0, 20);

    return ctx;
}

// Close CUDA miner and free resources
void cuda_miner_close(CUDAMinerContext* ctx) {
    if (!ctx) return;
    if (ctx->d_data_template) cudaFree(ctx->d_data_template);
    if (ctx->d_pattern) cudaFree(ctx->d_pattern);
    if (ctx->d_result_salt) cudaFree(ctx->d_result_salt);
    if (ctx->d_result_address) cudaFree(ctx->d_result_address);
    if (ctx->d_found) cudaFree(ctx->d_found);
    free(ctx);
}

// Run mining operation
// Returns: 1 if found, 0 if not found, -1 on error
int cuda_miner_mine(
    CUDAMinerContext* ctx,
    unsigned char* data_template,
    unsigned char* pattern,
    int pattern_length,
    unsigned long long start_nonce,
    unsigned char* result_salt,
    unsigned char* result_address
) {
    cudaError_t err;
    int found = 0;

    // Only copy template if changed (first 41 bytes are constant, rest changes with nonce in kernel)
    // Actually the template includes the salt prefix which is constant, so we can cache it
    if (!ctx->template_initialized || memcmp(ctx->cached_template, data_template, 85) != 0) {
        err = cudaMemcpy(ctx->d_data_template, data_template, 85, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -1;
        memcpy(ctx->cached_template, data_template, 85);
        ctx->template_initialized = 1;
    }

    // Only copy pattern if changed
    if (!ctx->pattern_initialized || ctx->cached_pattern_length != pattern_length ||
        memcmp(ctx->cached_pattern, pattern, pattern_length) != 0) {
        err = cudaMemcpy(ctx->d_pattern, pattern, pattern_length, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -1;
        memcpy(ctx->cached_pattern, pattern, pattern_length);
        ctx->cached_pattern_length = pattern_length;
        ctx->pattern_initialized = 1;
    }

    // Reset found flag using cudaMemsetAsync for better performance
    err = cudaMemset(ctx->d_found, 0, sizeof(int));
    if (err != cudaSuccess) return -1;

    // Launch kernel - 256 threads per block works best for this kernel
    // due to high register usage in keccak256
    int block_size = 256;
    int num_blocks = (ctx->batch_size + block_size - 1) / block_size;

    mine_create2<<<num_blocks, block_size>>>(
        ctx->d_data_template,
        ctx->d_pattern,
        pattern_length,
        (u64)start_nonce,
        ctx->d_result_salt,
        ctx->d_result_address,
        ctx->d_found
    );

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;

    // Check if found
    err = cudaMemcpy(&found, ctx->d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    if (found) {
        // Copy results back
        cudaMemcpy(result_salt, ctx->d_result_salt, 12, cudaMemcpyDeviceToHost);
        cudaMemcpy(result_address, ctx->d_result_address, 20, cudaMemcpyDeviceToHost);
        return 1;
    }

    return 0;
}

} // extern "C"

