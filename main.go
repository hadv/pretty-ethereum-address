package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"hash"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"golang.org/x/crypto/sha3"
)

const (
	// CREATE2 data buffer size: 0xff (1 byte) + deployer_address (20 bytes) + salt (32 bytes) + init_code_hash (32 bytes)
	size = 1 + 20 + 32 + 32
)

// hexCharToNibble converts a single hex character to its numeric value
func hexCharToNibble(c byte) byte {
	if c >= '0' && c <= '9' {
		return c - '0'
	}
	return c - 'a' + 10
}

// hexToByte converts two hex characters to a byte
func hexToByte(c1, c2 byte) byte {
	return (hexCharToNibble(c1) << 4) | hexCharToNibble(c2)
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

func main() {
	// Configuration - modify these values
	deployerAddress := common.HexToAddress("0x0000000000ffe8b47b3e2130213b802212439497")  // Replace with your deployer address
	initCodeHashStr := "747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a" // Replace with keccak256(init_code)
	pattern := "0x000000"                                                                 // Pattern to search for (prefix)
	numCores := runtime.NumCPU()                                                          // Use all available CPU cores (or set a specific number)

	// Set the maximum number of CPUs to use
	runtime.GOMAXPROCS(numCores)

	// Calculate number of goroutines (you can adjust the multiplier)
	numGoroutines := numCores * 800 // 800 goroutines per core

	initCodeHashBytes, err := hex.DecodeString(initCodeHashStr)
	if err != nil {
		fmt.Println("Error decoding init code hash:", err)
		os.Exit(1)
	}
	var initCodeHash [32]byte
	copy(initCodeHash[:], initCodeHashBytes)

	// Normalize pattern: remove 0x prefix and convert to lowercase
	normalizedPattern := strings.TrimPrefix(strings.ToLower(pattern), "0x")
	patternLen := len(normalizedPattern)

	// Pre-calculate pattern bytes once
	patternBytes := make([]byte, patternLen/2)
	for i := 0; i < patternLen/2; i++ {
		patternBytes[i] = hexToByte(normalizedPattern[i*2], normalizedPattern[i*2+1])
	}

	// Pre-calculate deployer address bytes once
	var deployerAddressBytes [20]byte
	copy(deployerAddressBytes[:], deployerAddress.Bytes())

	// Pre-calculate salt prefix (20 bytes) - only the last 12 bytes will be random
	saltPrefix := common.HexToAddress("0x18Ee4C040568238643C07e7aFd6c53efc196D26b")
	var saltPrefixBytes [20]byte
	copy(saltPrefixBytes[:], saltPrefix.Bytes())

	// Pre-build the CREATE2 data buffer template (only salt will change per iteration)
	// Format: 0xff ++ deployer_address ++ salt ++ init_code_hash
	dataTemplate := make([]byte, size)
	dataTemplate[0] = 0xff
	copy(dataTemplate[1:21], deployerAddressBytes[:])
	// bytes 21-53 are for salt (will be updated in each iteration)
	copy(dataTemplate[53:85], initCodeHash[:])

	fmt.Printf("Searching for CREATE2 address starting with '%s'...\n", pattern)
	fmt.Printf("Deployer: %s\n", deployerAddress.Hex())
	fmt.Printf("Init Code Hash: 0x%s\n", hex.EncodeToString(initCodeHash[:]))
	fmt.Printf("Using %d CPU cores with %d goroutines\n", numCores, numGoroutines)
	fmt.Println()

	// Start timing
	startTime := time.Now()

	quit := make(chan bool)
	var wg sync.WaitGroup
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Each goroutine gets its own data buffer to avoid race conditions
			data := append([]byte{}, dataTemplate...)

			// Create a reusable Keccak256 hasher for this goroutine
			hasher := sha3.NewLegacyKeccak256()

			for {
				select {
				case <-quit:
					return
				default:
					// Generate random salt with fixed prefix
					var salt [32]byte
					// Copy pre-calculated prefix (20 bytes)
					copy(salt[:20], saltPrefixBytes[:])
					// Random suffix (remaining 12 bytes)
					rand.Read(salt[20:])

					// Calculate CREATE2 address (returns bytes directly)
					addressBytes := calculateCreate2Address(hasher, data, salt)

					// Check if address matches pattern - direct byte comparison for performance
					matched := true
					for i := range patternBytes {
						if addressBytes[i] != patternBytes[i] {
							matched = false
							break
						}
					}

					if matched {
						elapsed := time.Since(startTime)
						fmt.Printf("\nFound!\n")
						fmt.Printf("Salt: 0x%s\n", hex.EncodeToString(salt[:]))
						fmt.Printf("Address: 0x%s\n", hex.EncodeToString(addressBytes[:]))
						fmt.Printf("Time elapsed: %s\n", elapsed)
						os.Exit(0)
					}
				}
			}
		}()
	}
	wg.Wait()
}
