//go:build !cuda
// +build !cuda

/*
 * Stub CUDA GPU Miner for builds without CUDA support
 * When CUDA is not available, these functions return appropriate errors
 */

package miner

import (
	"fmt"
	"time"
)

// CUDAGPUInfo contains information about a CUDA GPU device
type CUDAGPUInfo struct {
	Index        int
	Name         string
	ComputeUnits int
	MaxWorkSize  int
	TotalMemory  uint64
}

// CUDAMiner represents a CUDA GPU miner instance (stub)
type CUDAMiner struct{}

// ListCUDAGPUs returns an error when CUDA is not available
func ListCUDAGPUs() ([]CUDAGPUInfo, error) {
	return nil, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// NewCUDAMiner returns an error when CUDA is not available
func NewCUDAMiner(deviceIndex int, batchSize int) (*CUDAMiner, error) {
	return nil, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// Close is a no-op when CUDA is not available
func (m *CUDAMiner) Close() {}

// DeviceName returns empty string when CUDA is not available
func (m *CUDAMiner) DeviceName() string {
	return ""
}

// BatchSize returns 0 when CUDA is not available
func (m *CUDAMiner) BatchSize() int {
	return 0
}

// Mine returns an error when CUDA is not available
func (m *CUDAMiner) Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// MultiGPUMiner represents a Multi-GPU miner instance (stub)
type MultiGPUMiner struct{}

// NewMultiGPUMiner returns an error when CUDA is not available
func NewMultiGPUMiner(deviceIDs []int, batchSize int) (*MultiGPUMiner, error) {
	return nil, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// Close is a no-op
func (m *MultiGPUMiner) Close() {}

// DeviceNames returns empty list
func (m *MultiGPUMiner) DeviceNames() []string {
	return nil
}

// TotalBatchSize returns 0
func (m *MultiGPUMiner) TotalBatchSize() int {
	return 0
}

// Mine returns error
func (m *MultiGPUMiner) Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// EOAGPUResult contains the result of EOA mining (stub)
type EOAGPUResult struct {
	PrivateKey [32]byte
	Address    [20]byte
	Nonce      uint64
}

// EOACUDAMiner represents an EOA CUDA GPU miner instance (stub)
type EOACUDAMiner struct{}

// NewEOACUDAMiner returns an error when CUDA is not available
func NewEOACUDAMiner(deviceIndex int, batchSize int) (*EOACUDAMiner, error) {
	return nil, fmt.Errorf("CUDA EOA support not enabled. Build with: make build-cuda")
}

// Close is a no-op when CUDA is not available
func (m *EOACUDAMiner) Close() {}

// DeviceName returns empty string when CUDA is not available
func (m *EOACUDAMiner) DeviceName() string {
	return ""
}

// BatchSize returns 0 when CUDA is not available
func (m *EOACUDAMiner) BatchSize() int {
	return 0
}

// Mine returns an error when CUDA is not available
func (m *EOACUDAMiner) Mine(basePrivateKey []byte, pattern []byte, startNonce uint64) (*EOAGPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("CUDA EOA support not enabled. Build with: make build-cuda")
}

// EOACUDAMinerAvailable returns false when CUDA is not available
func EOACUDAMinerAvailable() bool {
	return false
}
