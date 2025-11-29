//go:build !darwin
// +build !darwin

/*
 * Stub GPU Miner for non-macOS platforms
 * GPU mining is only supported on macOS with OpenCL
 */

package miner

import (
	"fmt"
	"time"
)

// GPUMiner stub for non-Darwin platforms
type GPUMiner struct{}

// ListGPUs returns an error on non-Darwin platforms
func ListGPUs() ([]GPUInfo, error) {
	return nil, fmt.Errorf("GPU mining is only supported on macOS")
}

// NewGPUMiner returns an error on non-Darwin platforms
func NewGPUMiner(deviceIndex int, batchSize int) (*GPUMiner, error) {
	return nil, fmt.Errorf("GPU mining is only supported on macOS")
}

// Close is a no-op on non-Darwin platforms
func (m *GPUMiner) Close() {}

// DeviceName returns empty string on non-Darwin platforms
func (m *GPUMiner) DeviceName() string {
	return ""
}

// BatchSize returns 0 on non-Darwin platforms
func (m *GPUMiner) BatchSize() int {
	return 0
}

// Mine returns an error on non-Darwin platforms
func (m *GPUMiner) Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("GPU mining is only supported on macOS")
}

