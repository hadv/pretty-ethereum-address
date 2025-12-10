package miner

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cloudflare/circl/simd/keccakf1600"
	"github.com/decred/dcrd/dcrec/secp256k1/v4"
)

// EOASIMDMiner handles EOA-based vanity mining with SIMD acceleration
// Uses AVX2 4-way parallel Keccak and Sequential Derivation (P+G)
type EOASIMDMiner struct {
	numGoroutines int
	simdEnabled   bool
}

// NewEOASIMDMiner creates a new EOA miner
func NewEOASIMDMiner() *EOASIMDMiner {
	numCores := runtime.NumCPU()
	runtime.GOMAXPROCS(numCores)

	simdEnabled := keccakf1600.IsEnabledX4()

	return &EOASIMDMiner{
		numGoroutines: numCores * 4,
		simdEnabled:   simdEnabled,
	}
}

// EOAResult contains the found key and address
type EOAResult struct {
	PrivateKey  []byte
	Address     [20]byte
	Elapsed     time.Duration
	TotalHashes uint64
	HashRate    float64
}

// keccak256x4EOA computes 4 Keccak256 hashes of 64-byte public keys
// Input: 4 byte slices of 64 bytes each (X, Y concatenated)
// Output: 4 hash results of 32 bytes each
func keccak256x4EOA(perm *keccakf1600.StateX4, data [4][]byte, hashes *[4][32]byte) {
	state := perm.Initialize(false) // 24-round Keccak

	// Load 8 uint64 words (64 bytes)
	for lane := 0; lane < 4; lane++ {
		d := data[lane]
		for word := 0; word < 8; word++ {
			state[4*word+lane] = binary.LittleEndian.Uint64(d[word*8 : word*8+8])
		}

		// Padding: 0x01 at byte 64 (start of padding)
		// Byte 64 corresponds to the first byte of Word 8.
		state[4*8+lane] = 0x01
		
		// Zero out words 9..15 explicitly to handle reuse (Rate part)
		for word := 9; word < 16; word++ {
			state[4*word+lane] = 0
		}

		// Padding: 0x80 at byte 135 (end of block)
		// Byte 135 corresponds to the last byte (MSB) of Word 16.
		state[4*16+lane] = 0x8000000000000000
		
		// Zero out words 17..24 (Capacity part)
		for word := 17; word < 25; word++ {
			state[4*word+lane] = 0
		}
	}

	perm.Permute()

	// Extract generated hash (32 bytes = 4 words)
	for lane := 0; lane < 4; lane++ {
		for word := 0; word < 4; word++ {
			binary.LittleEndian.PutUint64(hashes[lane][word*8:word*8+8], state[4*word+lane])
		}
	}
}

// Generate random private key
func generateRandomKey() *big.Int {
	key, err := rand.Int(rand.Reader, secp256k1.S256().N)
	if err != nil {
		panic(err)
	}
	return key
}

func (m *EOASIMDMiner) Mine(patternStr string) *EOAResult {
	// Pattern cleaning
	if len(patternStr) >= 2 && patternStr[0:2] == "0x" {
		patternStr = patternStr[2:]
	}
	patternBytes, err := hex.DecodeString(patternStr)
	if err != nil {
		fmt.Printf("Error decoding pattern: %v\n", err)
		return nil
	}
	patternLen := len(patternBytes)

	startTime := time.Now()
	var found atomic.Bool
	var totalHashes atomic.Uint64
	var result *EOAResult
	var resultMu sync.Mutex

	// Progress reporter
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				hashes := totalHashes.Load()
				elapsed := time.Since(startTime)
				if elapsed.Seconds() > 0 {
					hashRate := float64(hashes) / elapsed.Seconds() / 1_000_000
					fmt.Printf("\rSearching... %d keys, %.2f MH/s", hashes, hashRate)
				}
			}
		}
	}()

	var wg sync.WaitGroup
	wg.Add(m.numGoroutines)

	// Constant Generator Point G
	Gx := secp256k1.S256().Gx
	Gy := secp256k1.S256().Gy

	for i := 0; i < m.numGoroutines; i++ {
		go func(threadID int) {
			defer wg.Done()
			
			// Create Jacobian Point for G (z=1) local to this thread
			GJac := &secp256k1.JacobianPoint{}
			GJac.X.SetByteSlice(Gx.Bytes())
			GJac.Y.SetByteSlice(Gy.Bytes())
			GJac.Z.SetInt(1).Normalize()

			// Initialize 4 lanes with random private keys
			privKeys := make([]*big.Int, 4)
			pubPoints := make([]*secp256k1.JacobianPoint, 4)

			for lane := 0; lane < 4; lane++ {
				privKeys[lane] = generateRandomKey()

				// Compute initial public point P = k*G
				x, y := secp256k1.S256().ScalarBaseMult(privKeys[lane].Bytes())
				pubPoints[lane] = &secp256k1.JacobianPoint{}
				pubPoints[lane].X.SetByteSlice(x.Bytes())
				pubPoints[lane].Y.SetByteSlice(y.Bytes())
				pubPoints[lane].Z.SetInt(1)
			}

			var perm keccakf1600.StateX4
			var hashes [4][32]byte

			// Reusable buffers for public key bytes (64 bytes each)
			pubKeyBytes := [4][]byte{
				make([]byte, 64), make([]byte, 64), make([]byte, 64), make([]byte, 64),
			}

			const checkInterval = 256
			counter := 0

			for {
				if found.Load() {
					return
				}

				if counter >= checkInterval {
					counter = 0
					totalHashes.Add(uint64(checkInterval * 4))
					if found.Load() {
						return
					}
				}
				counter++

				// 1. Montgomery Batch Inversion
				// Algorithm:
				// P0 = z0
				// P1 = z0 * z1
				// P2 = z0 * z1 * z2
				// P3 = z0 * z1 * z2 * z3
				// Inv = P3^-1
				// z3^-1 = P2 * Inv; Inv = Inv * z3
				// z2^-1 = P1 * Inv; Inv = Inv * z2
				// z1^-1 = P0 * Inv; Inv = Inv * z1
				// z0^-1 = Inv
				
				var p0, p1, p2, p3 secp256k1.FieldVal
				var inv secp256k1.FieldVal
				
				// Accumulate products
				p0.Set(&pubPoints[0].Z)
				p1.Mul2(&p0, &pubPoints[1].Z)
				p2.Mul2(&p1, &pubPoints[2].Z)
				p3.Mul2(&p2, &pubPoints[3].Z)
				
				// Single expensive inversion
				inv.Set(&p3)
				inv.Inverse() 
				
				// Back-propagate to find inverses of Z
				var zInv [4]secp256k1.FieldVal
				
				// z3^-1
				zInv[3].Mul2(&p2, &inv)
				inv.Mul(&pubPoints[3].Z) 
				
				// z2^-1
				zInv[2].Mul2(&p1, &inv)
				inv.Mul(&pubPoints[2].Z)
				
				// z1^-1
				zInv[1].Mul2(&p0, &inv)
				inv.Mul(&pubPoints[1].Z)
				
				// z0^-1
				zInv[0].Set(&inv)
				
				// Convert to Affine: x = X * (1/Z^2), y = Y * (1/Z^3)
				var zInv2, zInv3 secp256k1.FieldVal
				
				for lane := 0; lane < 4; lane++ {
					// z^2
					zInv2.SquareVal(&zInv[lane])
					// z^3
					zInv3.Mul2(&zInv2, &zInv[lane])
					
					// Update X, Y in place (treating them as affine now)
					// X = X * z^-2
					pubPoints[lane].X.Mul(&zInv2).Normalize()
					// Y = Y * z^-3
					pubPoints[lane].Y.Mul(&zInv3).Normalize()
					// Z = 1 (effectively)
					pubPoints[lane].Z.SetInt(1).Normalize() // Ensure normalized
					
					// Extract bytes
					pubPoints[lane].X.PutBytesUnchecked(pubKeyBytes[lane][0:32])
					pubPoints[lane].Y.PutBytesUnchecked(pubKeyBytes[lane][32:64])
				}

				// 2. Hash
				keccak256x4EOA(&perm, pubKeyBytes, &hashes)

				// 3. Check
				for lane := 0; lane < 4; lane++ {
					// Address is last 20 bytes of hash
					addr := hashes[lane][12:32]

					// Check pattern
					match := true
					for j := 0; j < patternLen; j++ {
						if addr[j] != patternBytes[j] {
							match = false
							break
						}
					}

					if match {
						if found.CompareAndSwap(false, true) {
							close(done)
							
							elapsed := time.Since(startTime)
							hashes := totalHashes.Load() + uint64(counter*4)
							hashRate := 0.0
							if elapsed.Seconds() > 0 {
								hashRate = float64(hashes) / elapsed.Seconds() / 1_000_000
							}
							
							var addrBytes [20]byte
							copy(addrBytes[:], addr)

							// Reconstruct Private Key
							foundKey := new(big.Int).Set(privKeys[lane])

							resultMu.Lock()
							result = &EOAResult{
								PrivateKey:  foundKey.Bytes(),
								Address:     addrBytes,
								Elapsed:     elapsed,
								TotalHashes: hashes,
								HashRate:    hashRate,
							}
							resultMu.Unlock()
						}
						return
					}
				}

				// 4. Update Step
				for lane := 0; lane < 4; lane++ {
					secp256k1.AddNonConst(pubPoints[lane], GJac, pubPoints[lane])
					privKeys[lane].Add(privKeys[lane], big.NewInt(1))
				}
			}
		}(i)
	}

	wg.Wait()
	return result
}
