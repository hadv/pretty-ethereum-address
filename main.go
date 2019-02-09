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
	var wg sync.WaitGroup
	wg.Add(5000)
	for i := 0; i < 5000; i++ {
		go func() {
			defer wg.Done()
			for {
				// Create an account
				key, _ := crypto.GenerateKey()
				// Get the address
				address := crypto.PubkeyToAddress(key.PublicKey).Hex()
				if strings.HasPrefix(address, "0x88888888") || strings.HasSuffix(address, "88888888") {
					// Get the private key
					privateKey := hex.EncodeToString(key.D.Bytes())
					fmt.Println(privateKey)
					os.Exit(0)
				}
			}
		}()
	}
	wg.Wait()
}
