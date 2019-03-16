package main

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/ethereum/go-ethereum/crypto"
)

func main() {
	quit := make(chan bool)
	var wg sync.WaitGroup
	for i := 0; i < 5000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-quit:
					os.Exit(0)
				default:
					// Create an account
					key, _ := crypto.GenerateKey()
					// Get the address
					address := crypto.PubkeyToAddress(key.PublicKey).Hex()
					if strings.HasSuffix(address, "888888") {
						// Get the private key
						privateKey := hex.EncodeToString(key.D.Bytes())
						fmt.Println(privateKey)
						quit <- true
					}
				}
			}
		}()
	}
	wg.Wait()
}
