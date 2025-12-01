.PHONY: build build-gpu clean run run-gpu install test help list-gpus

# Binary name
BINARY_NAME=vaneth

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

# Detect OS
UNAME_S := $(shell uname -s)

# Default target
all: build

## build: Build the binary with optimizations for speed
build:
	@echo "Building optimized binary..."
	GOPROXY=https://proxy.golang.org,direct $(GOBUILD) $(LDFLAGS) $(GCFLAGS) -o $(BINARY_NAME) -v

## build-gpu: Build with GPU support (macOS and Linux, uses OpenCL)
build-gpu:
	@echo "Building with GPU support..."
ifeq ($(UNAME_S),Linux)
	@echo "Detected Linux - using OpenCL with -lOpenCL"
endif
ifeq ($(UNAME_S),Darwin)
	@echo "Detected macOS - using OpenCL framework"
endif
	CGO_ENABLED=1 GOPROXY=https://proxy.golang.org,direct $(GOBUILD) $(LDFLAGS) $(GCFLAGS) -o $(BINARY_NAME) -v

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

## run: Build and run the application (CPU mode)
run: build
	@echo "Running in CPU mode..."
	./$(BINARY_NAME)

## run-gpu: Build and run in GPU mode (macOS and Linux)
run-gpu: build-gpu
	@echo "Running in GPU mode..."
	./$(BINARY_NAME) --gpu

## list-gpus: List available GPU devices
list-gpus: build-gpu
	./$(BINARY_NAME) --list-gpus

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
	@echo ""
	@echo "GPU Support (macOS and Linux):"
	@echo "  Uses OpenCL to accelerate mining on GPUs"
	@echo ""
	@echo "  macOS:"
	@echo "    Tested with: AMD Radeon Pro 555X"
	@echo "    Uses: -framework OpenCL"
	@echo ""
	@echo "  Linux:"
	@echo "    Supports: NVIDIA RTX 3000/4000/5000 series, AMD GPUs with ROCm"
	@echo "    Uses: -lOpenCL"
	@echo "    Prerequisites:"
	@echo "      sudo apt install opencl-headers ocl-icd-opencl-dev"
	@echo "      For NVIDIA: sudo apt install nvidia-opencl-dev"
	@echo ""
	@echo "Examples:"
	@echo "  make build-gpu && ./vaneth --list-gpus"
	@echo "  make run-gpu"

