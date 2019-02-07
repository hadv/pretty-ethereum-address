package main

import (
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/ethereum/go-ethereum/crypto"
)

func main() {
	for {
		// Create an account
		key, _ := crypto.GenerateKey()

		// Get the address
		address := crypto.PubkeyToAddress(key.PublicKey).Hex()
		fmt.Println(address)

		if strings.HasPrefix(address, "0x888888") {
			// Get the private key
			privateKey := hex.EncodeToString(key.D.Bytes())
			fmt.Println(privateKey)
			break
		}
	}
}
