//go:build cuda && linux
// +build cuda,linux

/*
 * CUDA GPU Miner for Linux with NVIDIA GPU support
 * Requires CUDA Toolkit 11.0+ and a GPU with Compute Capability 5.0+
 *
 * Build requirements:
 *   1. First compile the CUDA library: make build-cuda-lib
 *   2. Then build with: go build -tags cuda
 */

package miner

/*
#cgo LDFLAGS: -L${SRCDIR}/kernel -lvaneth_cuda -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm
#cgo CFLAGS: -I/usr/local/cuda/include

#include <stdlib.h>
#include <string.h>

// Device info structure (matches CUDA side)
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
    void* d_data_template;
    void* d_pattern;
    void* d_result_salt;
    void* d_result_address;
    int* d_found;
    int batch_size;
    char device_name[256];
} CUDAMinerContext;

// External functions from libvaneth_cuda.a
extern int get_cuda_device_count();
extern int get_cuda_device_info(int index, CUDADeviceInfo* info);
extern CUDAMinerContext* cuda_miner_init(int device_index, int batch_size);
extern void cuda_miner_close(CUDAMinerContext* ctx);
extern int cuda_miner_mine(
    CUDAMinerContext* ctx,
    unsigned char* data_template,
    unsigned char* pattern,
    int pattern_length,
    unsigned long long start_nonce,
    unsigned char* result_salt,
    unsigned char* result_address
);
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

// CUDAGPUInfo contains information about a CUDA GPU device
type CUDAGPUInfo struct {
	Index        int
	Name         string
	ComputeUnits int
	MaxWorkSize  int
	TotalMemory  uint64
}

// CUDAMiner represents a CUDA GPU miner instance
type CUDAMiner struct {
	ctx        *C.CUDAMinerContext
	deviceName string
	batchSize  int
}

// ListCUDAGPUs returns a list of available CUDA GPU devices
func ListCUDAGPUs() ([]CUDAGPUInfo, error) {
	count := int(C.get_cuda_device_count())
	if count <= 0 {
		// Negative values are CUDA error codes
		if count < 0 {
			cudaErr := -count
			// Common CUDA errors:
			// 35 = CUDA driver version insufficient
			// 38 = no CUDA-capable device
			// 100 = no device
			return nil, fmt.Errorf("CUDA error %d: check driver compatibility (kernel driver vs CUDA toolkit)", cudaErr)
		}
		return nil, fmt.Errorf("no CUDA devices found")
	}

	gpus := make([]CUDAGPUInfo, count)
	for i := 0; i < count; i++ {
		var info C.CUDADeviceInfo
		ret := C.get_cuda_device_info(C.int(i), &info)
		if ret != 0 {
			continue
		}
		gpus[i] = CUDAGPUInfo{
			Index:        int(info.index),
			Name:         C.GoString(&info.name[0]),
			ComputeUnits: int(info.compute_units),
			MaxWorkSize:  int(info.max_threads_per_block),
			TotalMemory:  uint64(info.total_memory),
		}
	}

	return gpus, nil
}

// NewCUDAMiner creates a new CUDA miner instance
func NewCUDAMiner(deviceIndex int, batchSize int) (*CUDAMiner, error) {
	ctx := C.cuda_miner_init(C.int(deviceIndex), C.int(batchSize))
	if ctx == nil {
		return nil, fmt.Errorf("failed to initialize CUDA miner on device %d", deviceIndex)
	}

	return &CUDAMiner{
		ctx:        ctx,
		deviceName: C.GoString(&ctx.device_name[0]),
		batchSize:  batchSize,
	}, nil
}

// Close releases all CUDA resources
func (m *CUDAMiner) Close() {
	if m.ctx != nil {
		C.cuda_miner_close(m.ctx)
		m.ctx = nil
	}
}

// DeviceName returns the name of the CUDA device
func (m *CUDAMiner) DeviceName() string {
	return m.deviceName
}

// BatchSize returns the current batch size
func (m *CUDAMiner) BatchSize() int {
	return m.batchSize
}

// Mine runs the CUDA mining operation
func (m *CUDAMiner) Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	startTime := time.Now()

	resultSalt := make([]byte, 12)
	resultAddress := make([]byte, 20)

	ret := C.cuda_miner_mine(
		m.ctx,
		(*C.uchar)(unsafe.Pointer(&dataTemplate[0])),
		(*C.uchar)(unsafe.Pointer(&pattern[0])),
		C.int(len(pattern)),
		C.ulonglong(startNonce),
		(*C.uchar)(unsafe.Pointer(&resultSalt[0])),
		(*C.uchar)(unsafe.Pointer(&resultAddress[0])),
	)

	elapsed := time.Since(startTime)

	if ret < 0 {
		return nil, elapsed, fmt.Errorf("CUDA mining error")
	}

	if ret == 1 {
		result := &GPUResult{}
		copy(result.SaltSuffix[:], resultSalt)
		copy(result.Address[:], resultAddress)
		result.Nonce = startNonce
		return result, elapsed, nil
	}

	return nil, elapsed, nil
}
