//go:build cuda && linux
// +build cuda,linux

package miner

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// MultiGPUMiner manages multiple CUDA miners
type MultiGPUMiner struct {
	miners    []*CUDAMiner
	deviceIDs []int
}

// NewMultiGPUMiner creates a new MultiGPUMiner instance
func NewMultiGPUMiner(deviceIDs []int, batchSize int) (*MultiGPUMiner, error) {
	if len(deviceIDs) == 0 {
		return nil, fmt.Errorf("no devices specified")
	}

	miners := make([]*CUDAMiner, 0, len(deviceIDs))

	for _, id := range deviceIDs {
		m, err := NewCUDAMiner(id, batchSize)
		if err != nil {
			// Cleanup already created miners
			for _, created := range miners {
				created.Close()
			}
			return nil, fmt.Errorf("failed to initialize miner on device %d: %v", id, err)
		}
		miners = append(miners, m)
	}

	return &MultiGPUMiner{
		miners:    miners,
		deviceIDs: deviceIDs,
	}, nil
}

// Close releases all resources
func (m *MultiGPUMiner) Close() {
	for _, miner := range m.miners {
		miner.Close()
	}
}

// DeviceNames returns the names of all used devices
func (m *MultiGPUMiner) DeviceNames() []string {
	names := make([]string, len(m.miners))
	for i, miner := range m.miners {
		names[i] = miner.DeviceName()
	}
	return names
}

// TotalBatchSize returns the combined batch size
func (m *MultiGPUMiner) TotalBatchSize() int {
	total := 0
	for _, miner := range m.miners {
		total += miner.BatchSize()
	}
	return total
}

// Mine runs the mining operation across all GPUs
func (m *MultiGPUMiner) Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	startTime := time.Now()
	
	resultChan := make(chan *GPUResult)
	errorChan := make(chan error)
	doneChan := make(chan struct{})
	
	var wg sync.WaitGroup
	var found atomic.Bool
	
	// Calculate stride - each GPU gets a slice of the nonce space
	// We'll use a simple approach: each GPU takes a step of (batchSize * numGPUs)
	// But since the CUDAMiner.Mine takes a startNonce and does one batch, 
	// we need to coordinate the start nonces for this iteration.
	
	// Actually, the CUDAMiner.Mine does ONE batch.
	// So we should just launch them all with different start nonces.
	// GPU 0: startNonce
	// GPU 1: startNonce + batchSize
	// ...
	
	for i, miner := range m.miners {
		wg.Add(1)
		go func(idx int, miner *CUDAMiner) {
			defer wg.Done()
			
			// Offset for this GPU
			gpuOffset := uint64(0)
			for j := 0; j < idx; j++ {
				gpuOffset += uint64(m.miners[j].BatchSize())
			}
			
			currentNonce := startNonce + gpuOffset
			
			res, _, err := miner.Mine(dataTemplate, pattern, currentNonce)
			if err != nil {
				select {
				case errorChan <- fmt.Errorf("device %d error: %v", m.deviceIDs[idx], err):
				default:
				}
				return
			}
			
			if res != nil {
				if found.CompareAndSwap(false, true) {
					select {
					case resultChan <- res:
					default:
					}
				}
			}
		}(i, miner)
	}
	
	// Wait for all to finish in a separate goroutine
	go func() {
		wg.Wait()
		close(doneChan)
	}()
	
	select {
	case res := <-resultChan:
		return res, time.Since(startTime), nil
	case err := <-errorChan:
		return nil, time.Since(startTime), err
	case <-doneChan:
		return nil, time.Since(startTime), nil
	}
}
