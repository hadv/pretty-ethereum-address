.PHONY: build build-gpu build-cuda build-cuda-lib clean run run-gpu run-cuda install test help list-gpus

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

# CUDA configuration
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_PATH ?= /usr/local/cuda
# Default to sm_80 (Ampere) for CUDA 12.x compatibility
# Override with: make build-cuda CUDA_ARCH=sm_XX
# Common architectures:
#   sm_70 - Volta (V100)
#   sm_75 - Turing (RTX 20xx)
#   sm_80 - Ampere (RTX 30xx, A100)
#   sm_89 - Ada Lovelace (RTX 40xx)
#   sm_90 - Hopper (H100)
CUDA_ARCH ?= sm_80

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

## build-cuda-lib: Compile the CUDA kernel library (required before build-cuda)
build-cuda-lib:
ifndef NVCC
	$(error "nvcc not found. Please install CUDA Toolkit 11.0+ and ensure nvcc is in PATH")
endif
	@echo "Compiling CUDA kernel library..."
	@echo "Using CUDA architecture: $(CUDA_ARCH)"
	cd miner/kernel && $(NVCC) -c -o cuda_miner.o cuda_launcher.cu -arch=$(CUDA_ARCH) -Xcompiler -fPIC
	cd miner/kernel && ar rcs libvaneth_cuda.a cuda_miner.o
	@echo "CUDA library built successfully: miner/kernel/libvaneth_cuda.a"

## build-cuda: Build with CUDA support for NVIDIA GPUs (Linux only)
build-cuda: build-cuda-lib
	@echo "Building with CUDA support..."
ifeq ($(UNAME_S),Linux)
	CGO_ENABLED=1 GOPROXY=https://proxy.golang.org,direct $(GOBUILD) -tags cuda $(LDFLAGS) $(GCFLAGS) -o $(BINARY_NAME) -v
else
	$(error "CUDA builds are only supported on Linux")
endif

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
	rm -f miner/kernel/*.o miner/kernel/*.a

## run: Build and run the application (CPU mode)
run: build
	@echo "Running in CPU mode..."
	./$(BINARY_NAME)

## run-gpu: Build and run in GPU mode (macOS and Linux, OpenCL)
run-gpu: build-gpu
	@echo "Running in GPU mode (OpenCL)..."
	./$(BINARY_NAME) --gpu

## run-cuda: Build and run in CUDA mode (Linux only)
run-cuda: build-cuda
	@echo "Running in GPU mode (CUDA)..."
	./$(BINARY_NAME) --gpu --gpu-backend cuda

## list-gpus: List available GPU devices
list-gpus: build-gpu
	./$(BINARY_NAME) --list-gpus

## list-cuda-gpus: List available CUDA GPU devices
list-cuda-gpus: build-cuda
	./$(BINARY_NAME) --list-gpus --gpu-backend cuda

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
	@echo "GPU Support:"
	@echo ""
	@echo "  OpenCL (macOS and Linux):"
	@echo "    Uses OpenCL to accelerate mining on GPUs"
	@echo ""
	@echo "    macOS:"
	@echo "      Tested with: AMD Radeon Pro 555X"
	@echo "      Uses: -framework OpenCL"
	@echo ""
	@echo "    Linux:"
	@echo "      Supports: NVIDIA RTX 3000/4000/5000 series, AMD GPUs with ROCm"
	@echo "      Uses: -lOpenCL"
	@echo "      Prerequisites:"
	@echo "        sudo apt install opencl-headers ocl-icd-opencl-dev"
	@echo "        For NVIDIA: sudo apt install nvidia-opencl-dev"
	@echo ""
	@echo "  CUDA (Linux only, NVIDIA GPUs):"
	@echo "    Uses CUDA for native NVIDIA GPU acceleration"
	@echo ""
	@echo "    Requirements:"
	@echo "      - NVIDIA CUDA Toolkit 11.0+"
	@echo "      - GPU with Compute Capability 5.0+ (Maxwell or newer)"
	@echo "      - nvcc compiler in PATH"
	@echo ""
	@echo "    Installation (Ubuntu/Debian):"
	@echo "      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
	@echo "      sudo dpkg -i cuda-keyring_1.1-1_all.deb"
	@echo "      sudo apt update && sudo apt install cuda-toolkit"
	@echo ""
	@echo "    Build and run:"
	@echo "      make build-cuda"
	@echo "      ./vaneth --gpu --gpu-backend cuda"
	@echo ""
	@echo "Examples:"
	@echo "  make build-gpu && ./vaneth --list-gpus"
	@echo "  make run-gpu"
	@echo "  make build-cuda && ./vaneth --gpu --gpu-backend cuda"

