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
- 🎯 **Flexible Patterns**: Search for any hex pattern (prefix matching)
- ⏱️ **Performance Tracking**: Shows time elapsed when a match is found
- 🔧 **Easy Configuration**: Simple variables to customize search parameters

## Installation

```bash
git clone https://github.com/hadv/pretty-ethereum-address.git
cd pretty-ethereum-address
go build
```

## Usage

### 1. Configure the Search Parameters

Edit `main.go` and set these values:

```go
deployerAddress := common.HexToAddress("0x18Ee4C040568238643C07e7aFd6c53efc196D26b")  // Your deployer address
initCodeHashStr := "ed6d47ef8858bf77ca8c43589269de4a0242b881ab9d2f8704546ce86ab20879" // keccak256(init_code)
pattern := "0x00000000"                                                               // Pattern to search for
numCores := runtime.NumCPU()                                                          // CPU cores to use
numGoroutines := numCores * 100                                                       // Goroutines per core
```

### 2. Get Your Init Code Hash

The init code hash is the keccak256 hash of your contract's creation bytecode. You can get this from:
- Hardhat/Foundry deployment scripts
- Remix compiler output
- Or calculate it manually: `keccak256(type(YourContract).creationCode)`

### 3. Run the Program

```bash
./pretty-ethereum-address
```

### 4. Example Output

```
Searching for CREATE2 address starting with '0x000000'...
Deployer: 0x18Ee4C040568238643C07e7aFd6c53efc196D26b
Init Code Hash: 0xed6d47ef8858bf77ca8c43589269de4a0242b881ab9d2f8704546ce86ab20879
Using 12 CPU cores with 9600 goroutines


Found!
Salt: 0xdc29329056cbe845d85b9877691d0932ea1a59fd37e878d7e6d0ae3464ebe397
Address: 0x000000f76928e94b2eced8d049b1772d54878cea
Time elapsed: 8.838849095s
```

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

- ✅ Direct byte comparison (no string allocations)
- ✅ Pre-calculated pattern and deployer bytes
- ✅ Returns bytes directly from CREATE2 calculation
- ✅ Early exit on byte mismatch
- ✅ Zero allocations in hot loop
- ✅ Multi-core parallel processing

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
