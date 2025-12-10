# Security Review: EOA Vanity Miner

## Scope
This review focuses on the randomness generation for private keys to ensure the implementation is not susceptible to seed-based brute-force attacks, such as the "Profanity" vulnerability (CVE-2022-39327).

## Findings

### 1. Entropy Source (Pass)
The miner uses the Go standard library's cryptographic random number generator:
```go
// miner/eoa_simd.go
func generateRandomKey() *big.Int {
	key, err := rand.Int(rand.Reader, secp256k1.S256().N)
    // ...
}
```
`crypto/rand.Reader` uses the OS-level CSPRNG (`/dev/urandom` on Unix-like systems, `CryptGenRandom` on Windows), providing high-quality entropy suitable for cryptographic keys.

### 2. Seed Size (Pass)
The generated private keys are full 256-bit integers (bounded by the curve order N), generated directly from the CSPRNG.
There is **no truncation** to 32-bit or 64-bit seeds. The search space for the initial key is $2^{256}$.

### 3. Independence (Pass)
Each mining "lane" (4 per goroutine) generates its own independent random start key:
```go
for lane := 0; lane < 4; lane++ {
    privKeys[lane] = generateRandomKey()
    // ...
}
```
This ensures that parallel workers do not share state or overlap in predictable ways.

### Comparison with "Profanity" Tool
| Feature | Profanity (Vulnerable) | Vaneth EOA Miner (Secure) |
| :--- | :--- | :--- |
| **Seed Source** | `rdtsc` / `random_device` (often 32-bit) | `crypto/rand` (CSPRNG) |
| **Seed Size** | 32-bit (brute-forceable in seconds) | 256-bit (secure) |
| **Key Generation** | Deterministic from small seed | Fully random |

## Conclusion
The EOA Vanity Miner implementation **IS NOT** vulnerable to the specific private key recovery attacks that affected the original Profanity tool. The use of a CSPRNG for the full 256-bit private key ensures that keys cannot be recovered by re-generating seeds.
