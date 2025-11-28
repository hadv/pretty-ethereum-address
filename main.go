package main

import (
	"crypto/rand"
	"encoding/hex"
	"flag"
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
	// Define CLI flags
	initCodeHashStr := flag.String("init-code-hash", "", "The keccak256 hash of the init code (hex string, required)")
	flag.StringVar(initCodeHashStr, "i", "", "The keccak256 hash of the init code (hex string, required) (shorthand)")

	pattern := flag.String("pattern", "0x00000000", "The address pattern/prefix to search for")
	flag.StringVar(pattern, "p", "0x00000000", "The address pattern/prefix to search for (shorthand)")

	saltPrefixStr := flag.String("salt-prefix", "", "The salt prefix address - first 20 bytes of salt (hex string, required)")
	flag.StringVar(saltPrefixStr, "s", "", "The salt prefix address - first 20 bytes of salt (hex string, required) (shorthand)")

	deployerAddressStr := flag.String("deployer", "0x0000000000ffe8b47b3e2130213b802212439497", "The deployer contract address")
	flag.StringVar(deployerAddressStr, "d", "0x0000000000ffe8b47b3e2130213b802212439497", "The deployer contract address (shorthand)")

	// Custom usage message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "A tool to find CREATE2 salt values that produce addresses with desired prefixes.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s -i 747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a -s 0x18Ee4C040568238643C07e7aFd6c53efc196D26b -p 0x00000000\n", os.Args[0])
	}

	flag.Parse()

	// Validate required flags
	if *initCodeHashStr == "" {
		fmt.Fprintln(os.Stderr, "Error: --init-code-hash (-i) is required")
		flag.Usage()
		os.Exit(1)
	}

	if *saltPrefixStr == "" {
		fmt.Fprintln(os.Stderr, "Error: --salt-prefix (-s) is required")
		flag.Usage()
		os.Exit(1)
	}

	// Validate and parse init code hash
	normalizedInitCodeHash := strings.TrimPrefix(strings.ToLower(*initCodeHashStr), "0x")
	if len(normalizedInitCodeHash) != 64 {
		fmt.Fprintln(os.Stderr, "Error: --init-code-hash must be a 32-byte hex string (64 hex characters)")
		os.Exit(1)
	}

	initCodeHashBytes, err := hex.DecodeString(normalizedInitCodeHash)
	if err != nil {
		fmt.Println("Error decoding init code hash:", err)
		os.Exit(1)
	}
	var initCodeHash [32]byte
	copy(initCodeHash[:], initCodeHashBytes)

	// Validate and parse salt prefix
	normalizedSaltPrefix := strings.TrimPrefix(strings.ToLower(*saltPrefixStr), "0x")
	if len(normalizedSaltPrefix) != 40 {
		fmt.Fprintln(os.Stderr, "Error: --salt-prefix must be a 20-byte address (40 hex characters)")
		os.Exit(1)
	}
	if !common.IsHexAddress(*saltPrefixStr) {
		fmt.Fprintln(os.Stderr, "Error: --salt-prefix is not a valid hex address")
		os.Exit(1)
	}
	saltPrefix := common.HexToAddress(*saltPrefixStr)

	// Validate and parse deployer address
	if !common.IsHexAddress(*deployerAddressStr) {
		fmt.Fprintln(os.Stderr, "Error: --deployer is not a valid hex address")
		os.Exit(1)
	}
	deployerAddress := common.HexToAddress(*deployerAddressStr)

	// Validate pattern
	normalizedPattern := strings.TrimPrefix(strings.ToLower(*pattern), "0x")
	if len(normalizedPattern)%2 != 0 {
		fmt.Fprintln(os.Stderr, "Error: --pattern must have an even number of hex characters")
		os.Exit(1)
	}
	if len(normalizedPattern) > 40 {
		fmt.Fprintln(os.Stderr, "Error: --pattern cannot be longer than 40 hex characters (20 bytes)")
		os.Exit(1)
	}
	for _, c := range normalizedPattern {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			fmt.Fprintln(os.Stderr, "Error: --pattern contains invalid hex characters")
			os.Exit(1)
		}
	}

	numCores := runtime.NumCPU() // Use all available CPU cores

	// Set the maximum number of CPUs to use
	runtime.GOMAXPROCS(numCores)

	// Calculate number of goroutines (you can adjust the multiplier)
	numGoroutines := numCores * 800 // 800 goroutines per core

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
	var saltPrefixBytes [20]byte
	copy(saltPrefixBytes[:], saltPrefix.Bytes())

	// Pre-build the CREATE2 data buffer template (only salt will change per iteration)
	// Format: 0xff ++ deployer_address ++ salt ++ init_code_hash
	dataTemplate := make([]byte, size)
	dataTemplate[0] = 0xff
	copy(dataTemplate[1:21], deployerAddressBytes[:])
	// bytes 21-53 are for salt (will be updated in each iteration)
	copy(dataTemplate[53:85], initCodeHash[:])

	fmt.Printf("Searching for CREATE2 address starting with '%s'...\n", *pattern)
	fmt.Printf("Deployer: %s\n", deployerAddress.Hex())
	fmt.Printf("Salt Prefix: %s\n", saltPrefix.Hex())
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

			// Pre-allocate salt with fixed prefix (only last 12 bytes change)
			var salt [32]byte
			copy(salt[:20], saltPrefixBytes[:])

		loop:
			for {
				select {
				case <-quit:
					return
				default:
					// Randomize only the last 12 bytes of salt
					rand.Read(salt[20:])

					// Calculate CREATE2 address (returns bytes directly)
					addressBytes := calculateCreate2Address(hasher, data, salt)

					// Check if address matches pattern - direct byte comparison for performance
					for i := range patternBytes {
						if addressBytes[i] != patternBytes[i] {
							continue loop
						}
					}

					// All bytes matched - we found it!
					elapsed := time.Since(startTime)
					fmt.Printf("\nFound!\n")
					fmt.Printf("Salt: 0x%s\n", hex.EncodeToString(salt[:]))
					fmt.Printf("Address: 0x%s\n", hex.EncodeToString(addressBytes[:]))
					fmt.Printf("Time elapsed: %s\n", elapsed)
					os.Exit(0)
				}
			}
		}()
	}
	wg.Wait()
}
