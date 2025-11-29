package miner

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"hash"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/crypto/sha3"
)

// CPUMiner handles CPU-based mining operations
type CPUMiner struct {
	numGoroutines int
}

// NewCPUMiner creates a new CPU miner
func NewCPUMiner() *CPUMiner {
	numCores := runtime.NumCPU()
	runtime.GOMAXPROCS(numCores)
	return &CPUMiner{
		numGoroutines: numCores * 800,
	}
}

// NumGoroutines returns the number of goroutines used for mining
func (m *CPUMiner) NumGoroutines() int {
	return m.numGoroutines
}

// NumCores returns the number of CPU cores
func (m *CPUMiner) NumCores() int {
	return runtime.NumCPU()
}

// calculateCreate2Address computes the CREATE2 address
// address = keccak256(0xff ++ deployer_address ++ salt ++ keccak256(init_code))[12:]
// Returns the 20-byte address directly
// The data buffer is reused and only the salt portion (bytes 21-53) needs to be updated
// The hasher is reused to avoid allocations in the hot loop
func calculateCreate2Address(hasher hash.Hash, data []byte, salt [32]byte) [20]byte {
	// Update only the salt portion (bytes 21-53)
	copy(data[21:53], salt[:])

	// Hash using reusable hasher
	hasher.Reset()
	hasher.Write(data)
	hashBytes := hasher.Sum(nil)

	// Take last 20 bytes for address
	var result [20]byte
	copy(result[:], hashBytes[12:32])
	return result
}

// Mine runs the CPU-based mining with goroutines
// Returns the result when found
func (m *CPUMiner) Mine(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte) *Result {
	startTime := time.Now()
	var found atomic.Bool

	var result *Result
	var resultMu sync.Mutex

	var wg sync.WaitGroup
	for range m.numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			data := append([]byte{}, dataTemplate...)
			hasher := sha3.NewLegacyKeccak256()

			var salt [32]byte
			copy(salt[:20], saltPrefixBytes[:])

		loop:
			for {
				if found.Load() {
					return
				}

				rand.Read(salt[20:])
				addressBytes := calculateCreate2Address(hasher, data, salt)

				for i := range patternBytes {
					if addressBytes[i] != patternBytes[i] {
						continue loop
					}
				}

				// Found it!
				if found.CompareAndSwap(false, true) {
					elapsed := time.Since(startTime)
					fmt.Printf("\nFound!\n")
					fmt.Printf("Salt: 0x%s\n", hex.EncodeToString(salt[:]))
					fmt.Printf("Address: 0x%s\n", hex.EncodeToString(addressBytes[:]))
					fmt.Printf("Time elapsed: %s\n", elapsed)

					resultMu.Lock()
					result = &Result{
						Salt:    salt,
						Address: addressBytes,
					}
					resultMu.Unlock()
				}
			}
		}()
	}
	wg.Wait()

	return result
}

