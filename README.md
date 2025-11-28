# pretty-ethereum-address

Generate vanity Ethereum contract addresses using CREATE2 opcode by finding the perfect salt.

## Overview

This tool uses brute force to find a salt value that, when used with the CREATE2 opcode, will deploy a smart contract to a vanity address (e.g., starting with `0x00000000`).

The CREATE2 address is calculated as:
```
address = keccak256(0xff ++ deployer_address ++ salt ++ keccak256(init_code))[12:]
```

## Features

- 🚀 **Highly Optimized**: Direct byte comparison, pre-calculated values, zero allocations in hot loop
- ⚡ **Multi-core Support**: Automatically uses all CPU cores with configurable goroutines
- 🎮 **GPU Acceleration**: OpenCL support for AMD GPUs on macOS (10-100x faster than CPU)
- 🎯 **Flexible Patterns**: Search for any hex pattern (prefix matching)
- 🔑 **Salt Prefix Support**: Generate salts with fixed prefixes for Solady CREATE2 factory
- ⏱️ **Performance Tracking**: Shows time elapsed when a match is found
- 🔧 **Easy Configuration**: CLI arguments for all search parameters

## Installation

### Standard Build (CPU only)

```bash
git clone https://github.com/hadv/pretty-ethereum-address.git
cd pretty-ethereum-address
go build
```

### Build with GPU Support (macOS)

For GPU acceleration on macOS with AMD GPUs:

```bash
make build-gpu
```

**Requirements for GPU build:**
- macOS with OpenCL framework (built-in)
- AMD GPU (e.g., Radeon Pro 555X, 560X, 5500M, etc.)
- CGO enabled (default on macOS)

## Usage

### Standard CREATE2 Deployment

#### 1. Configure the Search Parameters

Edit `main.go` and set these values:

```go
deployerAddress := common.HexToAddress("0x18Ee4C040568238643C07e7aFd6c53efc196D26b")  // Your deployer address
initCodeHashStr := "ed6d47ef8858bf77ca8c43589269de4a0242b881ab9d2f8704546ce86ab20879" // keccak256(init_code)
pattern := "0x00000000"                                                               // Pattern to search for
numCores := runtime.NumCPU()                                                          // CPU cores to use
numGoroutines := numCores * 100                                                       // Goroutines per core
```

#### 2. Get Your Init Code Hash

The init code hash is the keccak256 hash of your contract's creation bytecode. You can get this from:
- Hardhat/Foundry deployment scripts
- Remix compiler output
- Or calculate it manually: `keccak256(type(YourContract).creationCode)`

#### 3. Run the Program

```bash
./pretty-ethereum-address
```

#### 4. Example Output

```
Searching for CREATE2 address starting with '0x00000000'...
Deployer: 0x0000000000FFe8B47B3e2130213B802212439497
Init Code Hash: 0x747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a
Using 12 CPU cores with 9600 goroutines


Found!
Salt: 0x18ee4c040568238643c07e7afd6c53efc196d26bb3aa4a73c14310c5a4a12b1b
Address: 0x000000006e38ec9e8074ed84cbcbf4b9d8773b7e
Time elapsed: 2m51.724719347s
```

### Using Solady CREATE2 Factory

The Solady CREATE2 factory requires salts with a specific prefix (the caller's address). This tool supports generating salts with fixed prefixes.

#### Configuration for Solady Factory

The current configuration in `main.go` is set up for the Solady CREATE2 factory:

```go
// Solady CREATE2 Factory Configuration
deployerAddress := common.HexToAddress("0x0000000000ffe8b47b3e2130213b802212439497")  // Solady factory address
initCodeHashStr := "747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a" // Your contract's init code hash
pattern := "0x000000"                                                                 // Pattern to search for

// Salt prefix (pre-calculated outside the loop for performance)
saltPrefix := common.HexToAddress("0x18Ee4C040568238643C07e7aFd6c53efc196D26b")  // Your address (caller)
```

#### How Salt Prefix Works

- **First 20 bytes**: Fixed prefix (your address as the caller)
- **Last 12 bytes**: Randomly generated
- The prefix is pre-calculated once for optimal performance

#### Deploying with Solady Factory

Once you find a matching salt, deploy using the Solady CREATE2 factory:

```solidity
// Solady CREATE2 factory interface
interface ICreate2Factory {
    function deploy(bytes32 salt, bytes memory initCode) external returns (address);
}

// Deploy your contract
ICreate2Factory factory = ICreate2Factory(0x0000000000FFe8B47B3e2130213B802212439497);
bytes memory initCode = type(YourContract).creationCode;
bytes32 salt = 0x18Ee4C040568238643C07e7aFd6c53efc196D26b000000000000000000000123; // Salt found by the tool
address deployed = factory.deploy(salt, initCode);
```

#### Key Points

- The Solady factory is deployed at: `0x0000000000FFe8B47B3e2130213B802212439497`
- The factory automatically prepends the caller's address to the salt
- Your salt prefix must match your deploying address
- The tool optimizes by only randomizing the last 12 bytes

### GPU Mode (macOS)

GPU acceleration provides significantly faster mining by leveraging OpenCL on AMD GPUs.

#### List Available GPUs

```bash
./pretty-ethereum-address --list-gpus
```

Example output:
```
Available GPUs:
  [0] Intel(R) UHD Graphics 630
  [1] AMD Radeon Pro 555X Compute Engine
```

#### Run with GPU

```bash
./pretty-ethereum-address --gpu --gpu-device 1 \
  -i 747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a \
  -s 0x18Ee4C040568238643C07e7aFd6c53efc196D26b \
  -p 0x0000
```

#### GPU Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gpu`, `-g` | Enable GPU mode | `false` |
| `--gpu-device` | GPU device index | `0` |
| `--batch-size` | Hashes per GPU batch | `5000000` |
| `--list-gpus` | List available GPUs and exit | - |

#### GPU Example Output

```
Searching for CREATE2 address starting with '0x0000'...
Deployer: 0x0000000000FFe8B47B3e2130213B802212439497
Salt Prefix: 0x18Ee4C040568238643C07e7aFd6c53efc196D26b
Init Code Hash: 0x747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a
Using GPU: AMD Radeon Pro 555X Compute Engine
Batch size: 5000000 hashes per iteration

Found!
Salt: 0x18ee4c040568238643c07e7afd6c53efc196d26b000000000000000000004030
Address: 0x0000e628d423549be95a4113c4b59765b6cee09d
Time elapsed: 56.4974ms
Total hashes: 5000000
Hash rate: 88.50 MH/s
```

## Performance Comparison

### CPU vs GPU Performance

| Pattern | CPU Time | GPU Time | GPU Speedup |
|---------|----------|----------|-------------|
| `0x0000` (4 hex) | ~73ms | ~57ms | ~1.3x |
| `0x000000` (6 hex) | ~18s | ~1.5s | ~12x |
| `0x00000000` (8 hex) | ~5-8 min | ~30-60s | ~8-10x |

*Tested on MacBook Pro with AMD Radeon Pro 555X (12 CUs, 768 stream processors)*

### GPU Tips

- **Thermal Management**: MacBooks may throttle under heavy GPU load. Monitor temperatures.
- **Power**: Keep the laptop plugged in for best GPU performance.
- **Batch Size**: Larger batch sizes improve throughput but increase memory usage.
- **Device Selection**: Use `--list-gpus` to find your discrete GPU (usually index 1 on MacBooks).

## Performance Tuning

### CPU Cores
```go
// Use all available cores (default)
numCores := runtime.NumCPU()

// Limit to specific number
numCores := 8

// Use half of available cores
numCores := runtime.NumCPU() / 2
```

### Goroutines
```go
// Light load
numGoroutines := numCores * 10

// Balanced (recommended)
numGoroutines := numCores * 100

// Heavy load
numGoroutines := numCores * 1000
```

## Pattern Difficulty

The difficulty increases exponentially with pattern length:

| Pattern Length | Approximate Attempts | Time (estimate)    |
|---------------|---------------------|--------------------|
| 6 hex chars   | ~16 million         | seconds ~ minutes  |
| 8 hex chars   | ~4 billion          | minutes ~ hours    |
| 10 hex chars  | ~1 trillion         | hours ~ days       |

## How It Works

1. **Pre-calculation**: Pattern bytes and deployer address bytes are calculated once
2. **Parallel Search**: Multiple goroutines generate random salts concurrently
3. **CREATE2 Calculation**: Each salt is used to compute the resulting contract address
4. **Direct Comparison**: Bytes are compared directly (no string operations)
5. **Early Exit**: Stops immediately when a match is found

## Optimizations

### CPU Mode
- ✅ Direct byte comparison (no string allocations)
- ✅ Pre-calculated pattern and deployer bytes
- ✅ Pre-calculated salt prefix (for Solady factory mode)
- ✅ Returns bytes directly from CREATE2 calculation
- ✅ Early exit on byte mismatch
- ✅ Zero allocations in hot loop
- ✅ Multi-core parallel processing

### GPU Mode (macOS)
- ✅ OpenCL 1.2 compatible (native macOS support)
- ✅ Optimized Keccak256 kernel implementation
- ✅ Batch processing (millions of hashes per kernel launch)
- ✅ Atomic operations for early exit on match
- ✅ Efficient memory transfers between CPU and GPU
- ✅ Support for both Intel and AMD GPUs

## Use the Salt in Your Contract

Once you find a salt, use it in your Solidity contract:

```solidity
contract Factory {
    function deploy(bytes32 salt) public {
        bytes memory bytecode = type(YourContract).creationCode;
        address addr;
        assembly {
            addr := create2(0, add(bytecode, 0x20), mload(bytecode), salt)
        }
        require(addr != address(0), "Deploy failed");
    }
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Author

Ha ĐANG ([@hadv](https://github.com/hadv))
