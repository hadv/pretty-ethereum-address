.PHONY: build clean run install test help

# Binary name
BINARY_NAME=pretty-ethereum-address

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Build flags for optimization
LDFLAGS=-ldflags="-s -w"
GCFLAGS=-gcflags="-l=4"

# Default target
all: build

## build: Build the binary with optimizations for speed
build:
	@echo "Building optimized binary..."
	GOPROXY=https://proxy.golang.org,direct $(GOBUILD) $(LDFLAGS) $(GCFLAGS) -o $(BINARY_NAME) -v

## build-debug: Build the binary with debug symbols
build-debug:
	@echo "Building debug binary..."
	GOPROXY=https://proxy.golang.org,direct $(GOBUILD) -o $(BINARY_NAME) -v

## clean: Remove build artifacts
clean:
	@echo "Cleaning..."
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -rf dist/

## run: Build and run the application
run: build
	@echo "Running..."
	./$(BINARY_NAME)

## install: Install dependencies
install:
	@echo "Installing dependencies..."
	GOPROXY=https://proxy.golang.org,direct $(GOMOD) download
	GOPROXY=https://proxy.golang.org,direct $(GOMOD) tidy

## test: Run tests
test:
	@echo "Running tests..."
	$(GOTEST) -v ./...

## bench: Run benchmarks
bench:
	@echo "Running benchmarks..."
	$(GOTEST) -bench=. -benchmem ./...

## deps: Download and verify dependencies
deps:
	@echo "Downloading dependencies..."
	GOPROXY=https://proxy.golang.org,direct $(GOMOD) download
	@echo "Verifying dependencies..."
	$(GOMOD) verify

## tidy: Tidy up go.mod and go.sum
tidy:
	@echo "Tidying up dependencies..."
	$(GOMOD) tidy

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'

