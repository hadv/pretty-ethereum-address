package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// hexToByte converts two hex characters to a byte
func hexToByte(c1, c2 byte) byte {
	var b1, b2 byte
	if c1 >= '0' && c1 <= '9' {
		b1 = c1 - '0'
	} else if c1 >= 'a' && c1 <= 'f' {
		b1 = c1 - 'a' + 10
	}
	if c2 >= '0' && c2 <= '9' {
		b2 = c2 - '0'
	} else if c2 >= 'a' && c2 <= 'f' {
		b2 = c2 - 'a' + 10
	}
	return (b1 << 4) | b2
}

// calculateCreate2Address computes the CREATE2 address
// address = keccak256(0xff ++ deployer_address ++ salt ++ keccak256(init_code))[12:]
// Returns the 20-byte address directly
func calculateCreate2Address(deployerAddressBytes [20]byte, salt [32]byte, initCodeHash [32]byte) [20]byte {
	// Prepare the data: 0xff ++ deployer_address ++ salt ++ init_code_hash
	data := make([]byte, 1+20+32+32)
	data[0] = 0xff
	copy(data[1:21], deployerAddressBytes[:])
	copy(data[21:53], salt[:])
	copy(data[53:85], initCodeHash[:])

	// Hash and take last 20 bytes
	hash := crypto.Keccak256(data)
	var result [20]byte
	copy(result[:], hash[12:])
	return result
}

func main() {
	// Configuration - modify these values
	deployerAddress := common.HexToAddress("0x18Ee4C040568238643C07e7aFd6c53efc196D26b")  // Replace with your deployer address
	initCodeHashStr := "ed6d47ef8858bf77ca8c43589269de4a0242b881ab9d2f8704546ce86ab20879" // Replace with keccak256(init_code)
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

	fmt.Printf("Searching for CREATE2 address starting with '%s'...\n", pattern)
	fmt.Printf("Deployer: %s\n", deployerAddress.Hex())
	fmt.Printf("Init Code Hash: 0x%s\n", hex.EncodeToString(initCodeHash[:]))
	fmt.Printf("Using %d CPU cores with %d goroutines\n", numCores, numGoroutines)
	fmt.Println()

	// Start timing
	startTime := time.Now()

	quit := make(chan bool)
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-quit:
					return
				default:
					// Generate random salt
					var salt [32]byte
					rand.Read(salt[:])

					// Calculate CREATE2 address (returns bytes directly)
					addressBytes := calculateCreate2Address(deployerAddressBytes, salt, initCodeHash)

					// Check if address matches pattern - direct byte comparison for performance
					matched := true
					for i := 0; i < len(patternBytes); i++ {
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
