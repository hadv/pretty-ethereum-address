package main

import (
	cryptoRand "crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/hadv/vaneth/miner"
)

const logoCreate2 = `
__   _____    _   _ _____ _____ _   _
\ \ / / _ \  | \ | | ____|_   _| | | |
 \ V / |_| | |  \| |  _|   | | | |_| |
  \_/|_/ \_\ |_|\__|_____| |_| |_| |_|

      ⛏️  CREATE2 Vanity Miner  ⛏️
`

const logoEOA = `
__   _____    _   _ _____ _____ _   _
\ \ / / _ \  | \ | | ____|_   _| | | |
 \ V / |_| | |  \| |  _|   | | | |_| |
  \_/|_/ \_\ |_|\__|_____| |_| |_| |_|

       ⛏️  EOA Vanity Miner  ⛏️
`

func main() {
	// EOA mode flag (must check early)
	eoaMode := flag.Bool("eoa", false, "Mine EOA vanity addresses instead of CREATE2")

	// Define CLI flags
	initCodeHashStr := flag.String("init-code-hash", "", "The keccak256 hash of the init code (hex string, required for CREATE2)")
	flag.StringVar(initCodeHashStr, "i", "", "The keccak256 hash of the init code (hex string, required for CREATE2) (shorthand)")

	pattern := flag.String("pattern", "0x00000000", "The address pattern/prefix to search for")
	flag.StringVar(pattern, "p", "0x00000000", "The address pattern/prefix to search for (shorthand)")

	saltPrefixStr := flag.String("salt-prefix", "", "The salt prefix address - first 20 bytes of salt (hex string, required for CREATE2)")
	flag.StringVar(saltPrefixStr, "s", "", "The salt prefix address - first 20 bytes of salt (hex string, required for CREATE2) (shorthand)")

	deployerAddressStr := flag.String("deployer", "0x0000000000ffe8b47b3e2130213b802212439497", "The deployer contract address (CREATE2 only)")
	flag.StringVar(deployerAddressStr, "d", "0x0000000000ffe8b47b3e2130213b802212439497", "The deployer contract address (CREATE2 only) (shorthand)")

	// GPU flags
	useGPU := flag.Bool("gpu", false, "Use GPU for mining")
	flag.BoolVar(useGPU, "g", false, "Use GPU for mining (shorthand)")

	// SIMD flag
	useSIMD := flag.Bool("simd", false, "Use SIMD-optimized CPU miner (AVX2 4-way parallel Keccak)")

	gpuBackend := flag.String("gpu-backend", "opencl", "GPU backend to use: opencl, cuda, or auto")
	gpuDevice := flag.Int("gpu-device", 0, "GPU device index to use (deprecated, use --gpu-devices)")
	gpuDevicesStr := flag.String("gpu-devices", "", "GPU device indices to use (comma-separated or 'all'). Overrides --gpu-device")
	gpuStats := flag.Bool("gpu-stats", false, "Show per-GPU statistics during mining")
	batchSize := flag.Int("batch-size", 5000000, "Number of hashes per GPU batch (default 5M)")
	listGPUs := flag.Bool("list-gpus", false, "List available GPU devices and exit")

	// Custom usage message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "A tool to find vanity Ethereum addresses (CREATE2 or EOA).\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nCREATE2 Examples:\n")
		fmt.Fprintf(os.Stderr, "  CPU mode:\n")
		fmt.Fprintf(os.Stderr, "    %s -i 747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a -s 0x18Ee4C040568238643C07e7aFd6c53efc196D26b -p 0x00000000\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  GPU mode (CUDA):\n")
		fmt.Fprintf(os.Stderr, "    %s --gpu --gpu-backend cuda -i 747dd... -s 0x18Ee... -p 0x00000000\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "EOA Examples:\n")
		fmt.Fprintf(os.Stderr, "  EOA mode (CUDA GPU - finds wallet address with prefix):\n")
		fmt.Fprintf(os.Stderr, "    %s --eoa --gpu --gpu-backend cuda -p 0x00000000\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  EOA mode (multi-GPU):\n")
		fmt.Fprintf(os.Stderr, "    %s --eoa --gpu --gpu-backend cuda --gpu-devices all -p 0x00000000\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  List GPUs:\n")
		fmt.Fprintf(os.Stderr, "    %s --list-gpus --gpu-backend cuda\n", os.Args[0])
	}

	flag.Parse()

	// Handle --list-gpus
	if *listGPUs {
		backend := strings.ToLower(*gpuBackend)
		switch backend {
		case "cuda":
			cudaGPUs, err := miner.ListCUDAGPUs()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error listing CUDA GPUs: %v\n", err)
				os.Exit(1)
			}
			if len(cudaGPUs) == 0 {
				fmt.Println("No CUDA GPU devices found")
				os.Exit(0)
			}
			fmt.Printf("Found %d CUDA GPU device(s):\n\n", len(cudaGPUs))
			for _, gpu := range cudaGPUs {
				fmt.Printf("  Device %d: %s\n", gpu.Index, gpu.Name)
				fmt.Printf("    Compute Units (SMs): %d\n", gpu.ComputeUnits)
				fmt.Printf("    Max Threads per Block: %d\n", gpu.MaxWorkSize)
				fmt.Printf("    Total Memory: %d MB\n\n", gpu.TotalMemory/(1024*1024))
			}
		default: // opencl or auto
			gpus, err := miner.ListGPUs()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error listing GPUs: %v\n", err)
				os.Exit(1)
			}
			if len(gpus) == 0 {
				fmt.Println("No OpenCL GPU devices found")
				os.Exit(0)
			}
			fmt.Printf("Found %d OpenCL GPU device(s):\n\n", len(gpus))
			for _, gpu := range gpus {
				fmt.Printf("  Device %d: %s\n", gpu.Index, gpu.Name)
				fmt.Printf("    Vendor: %s\n", gpu.Vendor)
				fmt.Printf("    Compute Units: %d\n", gpu.ComputeUnits)
				fmt.Printf("    Max Work Group Size: %d\n\n", gpu.MaxWorkSize)
			}
		}
		os.Exit(0)
	}

	// Handle EOA mode
	if *eoaMode {
		fmt.Print(logoEOA)
		runEOAMode(*pattern, *useGPU, *gpuBackend, *gpuDevice, *gpuDevicesStr, *batchSize, *gpuStats)
		os.Exit(0)
	}

	// CREATE2 mode - print logo and validate required flags
	fmt.Print(logoCreate2)

	// Validate required flags for CREATE2
	if *initCodeHashStr == "" {
		fmt.Fprintln(os.Stderr, "Error: --init-code-hash (-i) is required for CREATE2 mode")
		flag.Usage()
		os.Exit(1)
	}

	if *saltPrefixStr == "" {
		fmt.Fprintln(os.Stderr, "Error: --salt-prefix (-s) is required for CREATE2 mode")
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
		patternBytes[i] = miner.HexToByte(normalizedPattern[i*2], normalizedPattern[i*2+1])
	}

	// Pre-calculate deployer address bytes once
	var deployerAddressBytes [20]byte
	copy(deployerAddressBytes[:], deployerAddress.Bytes())

	// Pre-calculate salt prefix (20 bytes) - only the last 12 bytes will be random
	var saltPrefixBytes [20]byte
	copy(saltPrefixBytes[:], saltPrefix.Bytes())

	// Pre-build the CREATE2 data buffer template (only salt will change per iteration)
	// Format: 0xff ++ deployer_address ++ salt ++ init_code_hash
	dataTemplate := make([]byte, miner.DataSize)
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
		backend := strings.ToLower(*gpuBackend)
		switch backend {
		case "cuda":
			runCUDAMiner(dataTemplate, patternBytes, saltPrefixBytes, *gpuDevice, *gpuDevicesStr, *batchSize, *gpuStats)
		case "auto":
			// Try CUDA first, fall back to OpenCL
			_, err := miner.ListCUDAGPUs()
			if err == nil {
				fmt.Println("Auto-detected CUDA backend")
				runCUDAMiner(dataTemplate, patternBytes, saltPrefixBytes, *gpuDevice, *gpuDevicesStr, *batchSize, *gpuStats)
			} else {
				fmt.Println("CUDA not available, using OpenCL backend")
				runGPUMiner(dataTemplate, patternBytes, saltPrefixBytes, *gpuDevice, *batchSize)
			}
		default: // opencl
			runGPUMiner(dataTemplate, patternBytes, saltPrefixBytes, *gpuDevice, *batchSize)
		}
	} else {
		runCPUMiner(dataTemplate, patternBytes, saltPrefixBytes, *useSIMD)
	}
}

// runGPUMiner runs the GPU-accelerated mining
func runGPUMiner(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte, deviceIndex int, batchSize int) {
	gpuMiner, err := miner.NewGPUMiner(deviceIndex, batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing GPU: %v\n", err)
		os.Exit(1)
	}
	defer gpuMiner.Close()

	fmt.Printf("Using GPU: %s\n", gpuMiner.DeviceName())
	fmt.Printf("Batch size: %d hashes per iteration\n", gpuMiner.BatchSize())
	fmt.Println()

	startTime := time.Now()
	var totalHashes uint64
	var nonce uint64

	for {
		result, batchTime, err := gpuMiner.Mine(dataTemplate, patternBytes, nonce)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GPU mining error: %v\n", err)
			os.Exit(1)
		}

		totalHashes += uint64(gpuMiner.BatchSize())
		nonce += uint64(gpuMiner.BatchSize())

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

// runCUDAMiner runs the CUDA GPU-accelerated mining
func runCUDAMiner(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte, deviceIndex int, devicesStr string, batchSize int, showStats bool) {
	var deviceIDs []int

	if devicesStr == "all" {
		gpus, err := miner.ListCUDAGPUs()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing CUDA GPUs: %v\n", err)
			os.Exit(1)
		}
		for _, gpu := range gpus {
			deviceIDs = append(deviceIDs, gpu.Index)
		}
	} else if devicesStr != "" {
		parts := strings.Split(devicesStr, ",")
		for _, part := range parts {
			id, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil {
				fmt.Fprintf(os.Stderr, "Invalid GPU device index: %s\n", part)
				os.Exit(1)
			}
			deviceIDs = append(deviceIDs, id)
		}
	} else {
		// Fallback to single device index
		deviceIDs = []int{deviceIndex}
	}

	multiMiner, err := miner.NewMultiGPUMiner(deviceIDs, batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing CUDA miner: %v\n", err)
		os.Exit(1)
	}
	defer multiMiner.Close()

	deviceNames := multiMiner.DeviceNames()
	fmt.Printf("Using %d CUDA GPU(s):\n", len(deviceNames))
	for i, name := range deviceNames {
		fmt.Printf("  GPU %d: %s\n", deviceIDs[i], name)
	}
	fmt.Printf("Total Batch size: %d hashes per iteration\n", multiMiner.TotalBatchSize())
	fmt.Println()

	startTime := time.Now()
	var totalHashes uint64
	var nonce uint64

	for {
		result, batchTime, err := multiMiner.Mine(dataTemplate, patternBytes, nonce)
		if err != nil {
			fmt.Fprintf(os.Stderr, "CUDA mining error: %v\n", err)
			os.Exit(1)
		}

		totalHashes += uint64(multiMiner.TotalBatchSize())
		nonce += uint64(multiMiner.TotalBatchSize())

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

			statsStr := ""
			if showStats {
				// TODO: Implement per-GPU stats if needed, for now just show combined
				// Since MultiGPUMiner doesn't return per-GPU stats in Mine(), we'd need to update it.
				// But for now, let's just show the combined stats.
			}

			fmt.Printf("\rSearching... %d hashes, %.2f MH/s, batch time: %v %s", totalHashes, hashRate, batchTime, statsStr)
		}
	}
}

// runCPUMiner runs the CPU-based mining with goroutines
func runCPUMiner(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte, useSIMD bool) {
	if useSIMD {
		simdMiner := miner.NewSIMDCPUMiner()
		if simdMiner.IsSIMDEnabled() {
			fmt.Printf("Using SIMD-optimized miner (AVX2 4-way parallel Keccak)\n")
			fmt.Printf("Using %d CPU cores with %d goroutines\n", simdMiner.NumCores(), simdMiner.NumGoroutines())
			fmt.Println()
			simdMiner.Mine(dataTemplate, patternBytes, saltPrefixBytes)
			os.Exit(0)
		} else {
			fmt.Printf("Warning: SIMD not available on this platform, falling back to standard CPU miner\n")
		}
	}

	cpuMiner := miner.NewCPUMiner()
	fmt.Printf("Using %d CPU cores with %d goroutines\n", cpuMiner.NumCores(), cpuMiner.NumGoroutines())
	fmt.Println()

	cpuMiner.Mine(dataTemplate, patternBytes, saltPrefixBytes)
	os.Exit(0)
}

// runEOAMode handles EOA vanity address mining
func runEOAMode(pattern string, useGPU bool, gpuBackend string, gpuDevice int, gpuDevicesStr string, batchSize int, gpuStats bool) {
	// Validate pattern
	normalizedPattern := strings.TrimPrefix(strings.ToLower(pattern), "0x")
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

	// Convert pattern to bytes
	patternBytes := make([]byte, len(normalizedPattern)/2)
	for i := 0; i < len(normalizedPattern)/2; i++ {
		patternBytes[i] = miner.HexToByte(normalizedPattern[i*2], normalizedPattern[i*2+1])
	}

	fmt.Printf("Mode: EOA (Externally Owned Account)\n")
	fmt.Printf("Searching for EOA address starting with '%s'...\n", pattern)
	fmt.Println()

	if !useGPU {
		fmt.Fprintln(os.Stderr, "Error: EOA mining currently requires GPU mode (--gpu --gpu-backend cuda)")
		fmt.Fprintln(os.Stderr, "CPU EOA mining is not yet implemented.")
		os.Exit(1)
	}

	backend := strings.ToLower(gpuBackend)
	if backend != "cuda" {
		fmt.Fprintln(os.Stderr, "Error: EOA mining currently only supports CUDA backend")
		fmt.Fprintln(os.Stderr, "Use: --gpu --gpu-backend cuda")
		os.Exit(1)
	}

	runEOACUDAMiner(patternBytes, gpuDevice, gpuDevicesStr, batchSize, gpuStats)
}

// runEOACUDAMiner runs the CUDA-accelerated EOA mining
func runEOACUDAMiner(patternBytes []byte, gpuDevice int, gpuDevicesStr string, batchSize int, gpuStats bool) {
	// gpuStats will be used in Phase 4 for per-GPU statistics
	_ = gpuStats

	var deviceIDs []int

	if gpuDevicesStr == "all" {
		gpus, err := miner.ListCUDAGPUs()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing CUDA GPUs: %v\n", err)
			os.Exit(1)
		}
		for _, gpu := range gpus {
			deviceIDs = append(deviceIDs, gpu.Index)
		}
	} else if gpuDevicesStr != "" {
		parts := strings.Split(gpuDevicesStr, ",")
		for _, part := range parts {
			id, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil {
				fmt.Fprintf(os.Stderr, "Invalid GPU device index: %s\n", part)
				os.Exit(1)
			}
			deviceIDs = append(deviceIDs, id)
		}
	} else {
		deviceIDs = []int{gpuDevice}
	}

	// For now, use single GPU EOA miner (multi-GPU EOA will be Phase 4)
	if len(deviceIDs) > 1 {
		fmt.Println("Note: Multi-GPU EOA mining uses the first GPU only (multi-GPU coming in Phase 4)")
	}

	eoaMiner, err := miner.NewEOACUDAMiner(deviceIDs[0], batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing EOA CUDA miner: %v\n", err)
		os.Exit(1)
	}
	defer eoaMiner.Close()

	fmt.Printf("Using GPU: %s\n", eoaMiner.DeviceName())
	fmt.Printf("Batch size: %d keys per iteration\n", eoaMiner.BatchSize())
	fmt.Println()

	// Generate random base private key
	basePrivateKey := make([]byte, 32)
	if _, err := cryptoRandRead(basePrivateKey); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating random private key: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("⚠️  SECURITY WARNING: Private keys will be displayed. Keep them secret!")
	fmt.Println()

	startTime := time.Now()
	var totalHashes uint64
	var nonce uint64

	for {
		result, batchTime, err := eoaMiner.Mine(basePrivateKey, patternBytes, nonce)
		if err != nil {
			fmt.Fprintf(os.Stderr, "EOA CUDA mining error: %v\n", err)
			os.Exit(1)
		}

		totalHashes += uint64(eoaMiner.BatchSize())
		nonce += uint64(eoaMiner.BatchSize())

		if result != nil {
			elapsed := time.Since(startTime)
			hashRate := float64(totalHashes) / elapsed.Seconds() / 1_000_000

			fmt.Printf("\n✅ Found EOA Vanity Address!\n\n")
			fmt.Printf("📍 Address:     0x%s\n", hex.EncodeToString(result.Address[:]))
			fmt.Printf("🔑 Private Key: 0x%s\n", hex.EncodeToString(result.PrivateKey[:]))
			fmt.Println()
			fmt.Printf("⚠️  Store your private key securely! Never share it with anyone.\n")
			fmt.Println()
			fmt.Printf("Time elapsed: %s\n", elapsed)
			fmt.Printf("Total keys checked: %d\n", totalHashes)
			fmt.Printf("Hash rate: %.2f MH/s\n", hashRate)
			os.Exit(0)
		}

		// Print progress every batch
		if nonce%uint64(batchSize) == 0 {
			elapsed := time.Since(startTime)
			hashRate := float64(totalHashes) / elapsed.Seconds() / 1_000_000
			fmt.Printf("\rSearching... %d keys, %.2f MH/s, batch time: %v", totalHashes, hashRate, batchTime)
		}
	}
}

// cryptoRandRead reads random bytes using crypto/rand
func cryptoRandRead(b []byte) (int, error) {
	return cryptoRand.Read(b)
}
