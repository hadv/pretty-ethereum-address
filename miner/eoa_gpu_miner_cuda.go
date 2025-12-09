//go:build cuda && linux
// +build cuda,linux

/*
 * CUDA GPU Miner for EOA Vanity Addresses
 * Requires CUDA Toolkit 11.0+ and a GPU with Compute Capability 5.0+
 *
 * Build requirements:
 *   1. First compile the CUDA library: make build-eoa-cuda-lib
 *   2. Then build with: go build -tags cuda
 */

package miner

/*
#cgo LDFLAGS: -L${SRCDIR}/kernel -lvaneth_eoa_cuda -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm
#cgo CFLAGS: -I/usr/local/cuda/include

#include <stdlib.h>
#include <string.h>

// EOA CUDA miner context (matches CUDA side)
typedef struct {
    int device_index;
    void* d_base_private_key;
    void* d_pattern;
    void* d_result_private_key;
    void* d_result_address;
    int* d_found;
    int batch_size;
    char device_name[256];
} EOACUDAMinerContext;

// External functions from libvaneth_eoa_cuda.a
extern EOACUDAMinerContext* eoa_cuda_miner_init(int device_index, int batch_size);
extern void eoa_cuda_miner_close(EOACUDAMinerContext* ctx);
extern int eoa_cuda_miner_mine(
    EOACUDAMinerContext* ctx,
    unsigned char* base_private_key,
    unsigned char* pattern,
    int pattern_length,
    unsigned long long start_nonce,
    unsigned char* batch_base_pub_x,
    unsigned char* batch_base_pub_y,
    unsigned char* result_private_key,
    unsigned char* result_address
);
*/
import "C"

import (
	"fmt"
	"math/big"
	"time"
	"unsafe"

	"github.com/ethereum/go-ethereum/crypto"
)

// EOAGPUResult contains the result of EOA mining
type EOAGPUResult struct {
	PrivateKey [32]byte
	Address    [20]byte
	Nonce      uint64
}

// EOACUDAMiner represents an EOA CUDA GPU miner instance
type EOACUDAMiner struct {
	ctx        *C.EOACUDAMinerContext
	deviceName string
	batchSize  int
}

// NewEOACUDAMiner creates a new EOA CUDA miner instance
func NewEOACUDAMiner(deviceIndex int, batchSize int) (*EOACUDAMiner, error) {
	ctx := C.eoa_cuda_miner_init(C.int(deviceIndex), C.int(batchSize))
	if ctx == nil {
		return nil, fmt.Errorf("failed to initialize EOA CUDA miner on device %d", deviceIndex)
	}

	return &EOACUDAMiner{
		ctx:        ctx,
		deviceName: C.GoString(&ctx.device_name[0]),
		batchSize:  batchSize,
	}, nil
}

// Close releases all CUDA resources
func (m *EOACUDAMiner) Close() {
	if m.ctx != nil {
		C.eoa_cuda_miner_close(m.ctx)
		m.ctx = nil
	}
}

// DeviceName returns the name of the CUDA device
func (m *EOACUDAMiner) DeviceName() string {
	return m.deviceName
}

// BatchSize returns the current batch size
func (m *EOACUDAMiner) BatchSize() int {
	return m.batchSize
}

// Mine runs the EOA CUDA mining operation
// Mine runs the EOA CUDA mining operation
func (m *EOACUDAMiner) Mine(basePrivateKey []byte, pattern []byte, startNonce uint64) (*EOAGPUResult, time.Duration, error) {
	if len(basePrivateKey) != 32 {
		return nil, 0, fmt.Errorf("base private key must be 32 bytes, got %d", len(basePrivateKey))
	}

	startTime := time.Now()

	// Calculate start private key: priv = base + nonce
	// Use big.Int for correct 256-bit addition
	privKeyBigInt := new(big.Int).SetBytes(basePrivateKey)
	privKeyBigInt.Add(privKeyBigInt, new(big.Int).SetUint64(startNonce))
	currentPrivKey := make([]byte, 32)
	privKeyBigInt.FillBytes(currentPrivKey)

	// Calculate Public Key for currentPrivKey uses go-ethereum/crypto
	// We need X and Y coordinates.
	// crypto.ToECDSA requires strict private key validity checks, but for random keys we assume valid.
	privKey, err := crypto.ToECDSA(currentPrivKey)
	if err != nil {
		return nil, 0, fmt.Errorf("invalid private key derived: %v", err)
	}

	// DEBUG: Print expected address for the batch base (idx=0)
	expectedAddr := crypto.PubkeyToAddress(privKey.PublicKey)
	fmt.Printf("DEBUG: Batch start addr (idx=0): %s\n", expectedAddr.Hex())

	// Get X and Y as 32-byte big-endian arrays
	pubX := privKey.PublicKey.X.Bytes()
	pubY := privKey.PublicKey.Y.Bytes()
	
	// Pad to 32 bytes if needed (big.Int.Bytes() strips leading zeros)
	pubXBytes := make([]byte, 32)
	pubYBytes := make([]byte, 32)
	copy(pubXBytes[32-len(pubX):], pubX)
	copy(pubYBytes[32-len(pubY):], pubY)

	resultPrivateKey := make([]byte, 32)
	resultAddress := make([]byte, 20)

	ret := C.eoa_cuda_miner_mine(
		m.ctx,
		(*C.uchar)(unsafe.Pointer(&basePrivateKey[0])),
		(*C.uchar)(unsafe.Pointer(&pattern[0])),
		C.int(len(pattern)),
		C.ulonglong(startNonce),
		(*C.uchar)(unsafe.Pointer(&pubXBytes[0])),
		(*C.uchar)(unsafe.Pointer(&pubYBytes[0])),
		(*C.uchar)(unsafe.Pointer(&resultPrivateKey[0])),
		(*C.uchar)(unsafe.Pointer(&resultAddress[0])),
	)

	elapsed := time.Since(startTime)

	if ret < 0 {
		return nil, elapsed, fmt.Errorf("EOA CUDA mining error")
	}

	if ret == 1 {
		result := &EOAGPUResult{}
		copy(result.PrivateKey[:], resultPrivateKey)
		copy(result.Address[:], resultAddress)
		result.Nonce = startNonce
		return result, elapsed, nil
	}

	return nil, elapsed, nil
}


// EOACUDAMinerAvailable returns true if CUDA EOA mining is available
func EOACUDAMinerAvailable() bool {
	gpus, err := ListCUDAGPUs()
	return err == nil && len(gpus) > 0
}

