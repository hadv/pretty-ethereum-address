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
	"sync/atomic"
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

	// GPU flags
	useGPU := flag.Bool("gpu", false, "Use GPU for mining (macOS only)")
	flag.BoolVar(useGPU, "g", false, "Use GPU for mining (macOS only) (shorthand)")

	gpuDevice := flag.Int("gpu-device", 0, "GPU device index to use")
	batchSize := flag.Int("batch-size", 5000000, "Number of hashes per GPU batch (default 5M)")
	listGPUs := flag.Bool("list-gpus", false, "List available GPU devices and exit")

	// Custom usage message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "A tool to find CREATE2 salt values that produce addresses with desired prefixes.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  CPU mode:\n")
		fmt.Fprintf(os.Stderr, "    %s -i 747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a -s 0x18Ee4C040568238643C07e7aFd6c53efc196D26b -p 0x00000000\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  GPU mode (macOS only):\n")
		fmt.Fprintf(os.Stderr, "    %s --gpu -i 747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a -s 0x18Ee4C040568238643C07e7aFd6c53efc196D26b -p 0x00000000\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  List GPUs:\n")
		fmt.Fprintf(os.Stderr, "    %s --list-gpus\n", os.Args[0])
	}

	flag.Parse()

	// Handle --list-gpus
	if *listGPUs {
		gpus, err := ListGPUs()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing GPUs: %v\n", err)
			os.Exit(1)
		}
		if len(gpus) == 0 {
			fmt.Println("No GPU devices found")
			os.Exit(0)
		}
		fmt.Printf("Found %d GPU device(s):\n\n", len(gpus))
		for _, gpu := range gpus {
			fmt.Printf("  Device %d: %s\n", gpu.Index, gpu.Name)
			fmt.Printf("    Vendor: %s\n", gpu.Vendor)
			fmt.Printf("    Compute Units: %d\n", gpu.ComputeUnits)
			fmt.Printf("    Max Work Group Size: %d\n\n", gpu.MaxWorkSize)
		}
		os.Exit(0)
	}

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
	// bytes 21-40 are for salt prefix (first 20 bytes)
	copy(dataTemplate[21:41], saltPrefixBytes[:])
	// bytes 41-53 are for salt suffix (last 12 bytes) - will be updated in each iteration
	copy(dataTemplate[53:85], initCodeHash[:])

	fmt.Printf("Searching for CREATE2 address starting with '%s'...\n", *pattern)
	fmt.Printf("Deployer: %s\n", deployerAddress.Hex())
	fmt.Printf("Salt Prefix: %s\n", saltPrefix.Hex())
	fmt.Printf("Init Code Hash: 0x%s\n", hex.EncodeToString(initCodeHash[:]))

	// Branch based on GPU or CPU mode
	if *useGPU {
		runGPUMiner(dataTemplate, patternBytes, saltPrefixBytes, *gpuDevice, *batchSize)
	} else {
		runCPUMiner(dataTemplate, patternBytes, saltPrefixBytes)
	}
}

// runGPUMiner runs the GPU-accelerated mining
func runGPUMiner(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte, deviceIndex int, batchSize int) {
	miner, err := NewGPUMiner(deviceIndex, batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing GPU: %v\n", err)
		os.Exit(1)
	}
	defer miner.Close()

	fmt.Printf("Using GPU: %s\n", miner.DeviceName())
	fmt.Printf("Batch size: %d hashes per iteration\n", miner.BatchSize())
	fmt.Println()

	startTime := time.Now()
	var totalHashes uint64
	var nonce uint64

	for {
		result, batchTime, err := miner.Mine(dataTemplate, patternBytes, nonce)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GPU mining error: %v\n", err)
			os.Exit(1)
		}

		totalHashes += uint64(miner.BatchSize())
		nonce += uint64(miner.BatchSize())

		if result != nil {
			// Construct the full salt
			var salt [32]byte
			copy(salt[:20], saltPrefixBytes[:])
			copy(salt[20:], result.SaltSuffix[:])

			elapsed := time.Since(startTime)
			hashRate := float64(totalHashes) / elapsed.Seconds() / 1_000_000

			fmt.Printf("\nFound!\n")
			fmt.Printf("Salt: 0x%s\n", hex.EncodeToString(salt[:]))
			fmt.Printf("Address: 0x%s\n", hex.EncodeToString(result.Address[:]))
			fmt.Printf("Time elapsed: %s\n", elapsed)
			fmt.Printf("Total hashes: %d\n", totalHashes)
			fmt.Printf("Hash rate: %.2f MH/s\n", hashRate)
			os.Exit(0)
		}

		// Print progress every ~10 batches
		if nonce%(uint64(batchSize)*10) == 0 {
			elapsed := time.Since(startTime)
			hashRate := float64(totalHashes) / elapsed.Seconds() / 1_000_000
			fmt.Printf("\rSearching... %d hashes, %.2f MH/s, batch time: %v", totalHashes, hashRate, batchTime)
		}
	}
}

// runCPUMiner runs the CPU-based mining with goroutines
func runCPUMiner(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte) {
	numCores := runtime.NumCPU()
	runtime.GOMAXPROCS(numCores)
	numGoroutines := numCores * 800

	fmt.Printf("Using %d CPU cores with %d goroutines\n", numCores, numGoroutines)
	fmt.Println()

	startTime := time.Now()
	var found atomic.Bool

	var wg sync.WaitGroup
	for range numGoroutines {
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
					os.Exit(0)
				}
			}
		}()
	}
	wg.Wait()
}
