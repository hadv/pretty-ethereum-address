// Package miner provides CPU and GPU mining capabilities for CREATE2 vanity address generation.
package miner

import "time"

const (
	// DataSize is the CREATE2 data buffer size:
	// 0xff (1 byte) + deployer_address (20 bytes) + salt (32 bytes) + init_code_hash (32 bytes)
	DataSize = 1 + 20 + 32 + 32
)

// Result contains the result of a successful mining operation
type Result struct {
	Salt    [32]byte
	Address [20]byte
	Nonce   uint64
}

// GPUInfo contains information about a GPU device
type GPUInfo struct {
	Index        int
	Name         string
	Vendor       string
	ComputeUnits int
	MaxWorkSize  int
}

// GPUResult contains the result from GPU mining
type GPUResult struct {
	SaltSuffix [12]byte
	Address    [20]byte
	Nonce      uint64
}

// GPUMinerInterface defines the interface for GPU miners
type GPUMinerInterface interface {
	// Mine runs the GPU mining operation
	// dataTemplate: 85-byte CREATE2 data with salt prefix already set (bytes 21-40)
	// pattern: The address pattern to match
	// startNonce: Starting nonce for this batch
	// Returns the result and elapsed time, or nil if nothing found in this batch
	Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error)

	// Close releases all resources
	Close()

	// DeviceName returns the name of the GPU device
	DeviceName() string

	// BatchSize returns the current batch size
	BatchSize() int
}

// HexCharToNibble converts a single hex character to its numeric value
func HexCharToNibble(c byte) byte {
	if c >= '0' && c <= '9' {
		return c - '0'
	}
	return c - 'a' + 10
}

// HexToByte converts two hex characters to a byte
func HexToByte(c1, c2 byte) byte {
	return (HexCharToNibble(c1) << 4) | HexCharToNibble(c2)
}

