/*
 * CUDA Library for EOA Vanity Address Mining
 * Provides C-callable functions for the Go bindings
 *
 * Compile with: nvcc -c -o eoa_miner.o eoa_launcher.cu -arch=sm_50
 *               ar rcs libvaneth_eoa_cuda.a eoa_miner.o
 */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Include the kernel implementation
#include "eoa_miner.cu"

// EOA CUDA miner context
typedef struct {
    int device_index;
    unsigned char* d_result_private_key;
    unsigned char* d_result_address;
    int* d_found;
    void* d_generator_table; // AffinePoint*
    void* d_batch_pub_x;
    void* d_batch_pub_y;
    int batch_size;
    char device_name[256];
    // Cache for avoiding repeated copies to constant memory
    unsigned char cached_private_key[32];
    unsigned char cached_pattern[20];
    int cached_pattern_length;
    int privkey_initialized;
    int pattern_initialized;
    int table_initialized;
} EOACUDAMinerContext;

extern "C" {

// Initialize EOA CUDA miner
EOACUDAMinerContext* eoa_cuda_miner_init(int device_index, int batch_size) {
    cudaError_t err;

    // Set device
    err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
        return NULL;
    }

    EOACUDAMinerContext* ctx = (EOACUDAMinerContext*)malloc(sizeof(EOACUDAMinerContext));
    if (!ctx) return NULL;

    ctx->device_index = device_index;
    ctx->batch_size = batch_size;

    // Get device name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    strncpy(ctx->device_name, prop.name, 255);
    ctx->device_name[255] = '\0';

    // Allocate result memory
    err = cudaMalloc(&ctx->d_result_private_key, 32);
    if (err != cudaSuccess) { free(ctx); return NULL; }

    err = cudaMalloc(&ctx->d_result_address, 20);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_result_private_key);
        free(ctx); return NULL;
    }

    err = cudaMalloc(&ctx->d_found, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_result_private_key);
        cudaFree(ctx->d_result_address);
        free(ctx); return NULL;
    }

    // Allocate generator table (BatchSize * 64 bytes)
    // AffinePoint is 64 bytes
    size_t table_size = (size_t)batch_size * 64; 
    err = cudaMalloc(&ctx->d_generator_table, table_size);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_result_private_key);
        cudaFree(ctx->d_result_address);
        cudaFree(ctx->d_found);
        free(ctx); return NULL;
    }
    
    // Allocate batch base pubkeys (Global Memory approach)
    err = cudaMalloc(&ctx->d_batch_pub_x, 32);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_generator_table);
        // ... (cleanup others)
        free(ctx); return NULL;
    }
    err = cudaMalloc(&ctx->d_batch_pub_y, 32);
    if (err != cudaSuccess) {
        cudaFree(ctx->d_batch_pub_x);
        cudaFree(ctx->d_generator_table);
        // ...
        free(ctx); return NULL;
    }

    // Initialize generator table
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    init_generator_table<<<num_blocks, block_size>>>((AffinePoint*)ctx->d_generator_table, batch_size);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
         // Cleanup...
         cudaFree(ctx->d_batch_pub_x);
         cudaFree(ctx->d_batch_pub_y);
         cudaFree(ctx->d_generator_table);
         cudaFree(ctx->d_result_private_key);
         cudaFree(ctx->d_result_address);
         cudaFree(ctx->d_found);
         free(ctx); return NULL;
    }

    // Initialize cache flags
    ctx->privkey_initialized = 0;
    ctx->pattern_initialized = 0;
    ctx->cached_pattern_length = 0;
    ctx->table_initialized = 1;
    memset(ctx->cached_private_key, 0, 32);
    memset(ctx->cached_pattern, 0, 20);

    return ctx;
}

// Close EOA CUDA miner and free resources
void eoa_cuda_miner_close(EOACUDAMinerContext* ctx) {
    if (!ctx) return;

    // Set device to ensure we free the correct resources
    cudaSetDevice(ctx->device_index);

    if (ctx->d_result_private_key) cudaFree(ctx->d_result_private_key);
    if (ctx->d_result_address) cudaFree(ctx->d_result_address);
    if (ctx->d_found) cudaFree(ctx->d_found);
    if (ctx->d_generator_table) cudaFree(ctx->d_generator_table);
    free(ctx);
}

// Run EOA mining operation
// Returns: 1 if found, 0 if not found, -1 on error
int eoa_cuda_miner_mine(
    EOACUDAMinerContext* ctx,
    unsigned char* base_private_key,
    unsigned char* pattern,
    int pattern_length,
    unsigned long long start_nonce,
    unsigned char* batch_base_pub_x, // 32 bytes
    unsigned char* batch_base_pub_y, // 32 bytes
    unsigned char* result_private_key,
    unsigned char* result_address
) {
    cudaError_t err;
    int found = 0;

    // Set device for this thread
    err = cudaSetDevice(ctx->device_index);
    if (err != cudaSuccess) return -1;

    // Copy private key to constant memory (with caching)
    if (!ctx->privkey_initialized || memcmp(ctx->cached_private_key, base_private_key, 32) != 0) {
        err = cudaMemcpyToSymbol(c_base_private_key, base_private_key, 32);
        if (err != cudaSuccess) {
             printf("CUDA error copying private key: %d (%s)\n", err, cudaGetErrorString(err));
             return -1;
        }
        memcpy(ctx->cached_private_key, base_private_key, 32);
        ctx->privkey_initialized = 1;
    }

    // Copy pattern to constant memory (with caching)
    if (!ctx->pattern_initialized || ctx->cached_pattern_length != pattern_length ||
        memcmp(ctx->cached_pattern, pattern, pattern_length) != 0) {
        err = cudaMemcpyToSymbol(c_pattern, pattern, pattern_length);
        if (err != cudaSuccess) {
             printf("CUDA error copying pattern: %d (%s)\n", err, cudaGetErrorString(err));
             return -1;
        }
        
        err = cudaMemcpyToSymbol(c_pattern_length, &pattern_length, sizeof(int));
        if (err != cudaSuccess) {
             printf("CUDA error copying pattern length: %d (%s)\n", err, cudaGetErrorString(err));
             return -1;
        }

        memcpy(ctx->cached_pattern, pattern, pattern_length);
        ctx->cached_pattern_length = pattern_length;
        ctx->pattern_initialized = 1;
    }

    // Reset found flag
    err = cudaMemset(ctx->d_found, 0, sizeof(int));
    if (err != cudaSuccess) return -1;

    // Copy batch base public key to constant memory
    // Use cudaGetSymbolAddress + cudaMemcpy for explicit handling
    void* d_pub_x = NULL;
    void* d_pub_y = NULL;
    
    err = cudaGetSymbolAddress(&d_pub_x, c_batch_base_pub_x);
    if (err != cudaSuccess) {
         printf("CUDA error getting symbol address x: %d (%s)\n", err, cudaGetErrorString(err));
         return -1;
    }
    err = cudaMemcpy(d_pub_x, batch_base_pub_x, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         printf("CUDA error copying batch base pub x: %d (%s)\n", err, cudaGetErrorString(err));
         return -1;
    }

    err = cudaGetSymbolAddress(&d_pub_y, c_batch_base_pub_y);
    if (err != cudaSuccess) {
         printf("CUDA error getting symbol address y: %d (%s)\n", err, cudaGetErrorString(err));
         return -1;
    }
    err = cudaMemcpy(d_pub_y, batch_base_pub_y, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         printf("CUDA error copying batch base pub y: %d (%s)\n", err, cudaGetErrorString(err));
         return -1;
    }

    // Launch kernel - 256 threads per block
    int block_size = 256;
    int num_blocks = (ctx->batch_size + block_size - 1) / block_size;

    mine_eoa_opt<<<num_blocks, block_size>>>(
        (u64)start_nonce,
        (AffinePoint*)ctx->d_generator_table,
        ctx->d_result_private_key,
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
        cudaMemcpy(result_private_key, ctx->d_result_private_key, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(result_address, ctx->d_result_address, 20, cudaMemcpyDeviceToHost);
        return 1;
    }

    return 0;
}

} // extern "C"

