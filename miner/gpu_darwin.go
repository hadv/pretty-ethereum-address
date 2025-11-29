//go:build darwin
// +build darwin

/*
 * GPU Miner for macOS with AMD Radeon Pro 555X
 * Uses OpenCL 1.2 for cross-platform GPU acceleration
 */

package miner

/*
#cgo darwin LDFLAGS: -framework OpenCL
#include <OpenCL/opencl.h>
#include <stdlib.h>
#include <string.h>

// Helper to get error string
const char* cl_error_string(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        default: return "Unknown error";
    }
}
*/
import "C"

import (
	"embed"
	"fmt"
	"time"
	"unsafe"
)

//go:embed kernel/keccak256.cl
var kernelFS embed.FS

// GPUMiner represents an OpenCL GPU miner instance
type GPUMiner struct {
	platform    C.cl_platform_id
	device      C.cl_device_id
	context     C.cl_context
	queue       C.cl_command_queue
	program     C.cl_program
	kernel      C.cl_kernel
	deviceName  string
	maxWorkSize int
	batchSize   int
}

// ListGPUs returns a list of available GPU devices
func ListGPUs() ([]GPUInfo, error) {
	var numPlatforms C.cl_uint
	ret := C.clGetPlatformIDs(0, nil, &numPlatforms)
	if ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platform count: %s", C.GoString(C.cl_error_string(ret)))
	}

	if numPlatforms == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found")
	}

	platforms := make([]C.cl_platform_id, numPlatforms)
	ret = C.clGetPlatformIDs(numPlatforms, &platforms[0], nil)
	if ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platforms: %s", C.GoString(C.cl_error_string(ret)))
	}

	var gpus []GPUInfo
	gpuIndex := 0

	for _, platform := range platforms {
		var numDevices C.cl_uint
		ret = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if ret != C.CL_SUCCESS || numDevices == 0 {
			continue
		}

		devices := make([]C.cl_device_id, numDevices)
		ret = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, numDevices, &devices[0], nil)
		if ret != C.CL_SUCCESS {
			continue
		}

		for _, device := range devices {
			info := getDeviceInfo(device)
			info.Index = gpuIndex
			gpus = append(gpus, info)
			gpuIndex++
		}
	}

	return gpus, nil
}

func getDeviceInfo(device C.cl_device_id) GPUInfo {
	var info GPUInfo

	// Get device name
	var nameSize C.size_t
	C.clGetDeviceInfo(device, C.CL_DEVICE_NAME, 0, nil, &nameSize)
	nameBuf := make([]byte, nameSize)
	C.clGetDeviceInfo(device, C.CL_DEVICE_NAME, nameSize, unsafe.Pointer(&nameBuf[0]), nil)
	info.Name = string(nameBuf[:nameSize-1])

	// Get vendor
	var vendorSize C.size_t
	C.clGetDeviceInfo(device, C.CL_DEVICE_VENDOR, 0, nil, &vendorSize)
	vendorBuf := make([]byte, vendorSize)
	C.clGetDeviceInfo(device, C.CL_DEVICE_VENDOR, vendorSize, unsafe.Pointer(&vendorBuf[0]), nil)
	info.Vendor = string(vendorBuf[:vendorSize-1])

	// Get compute units
	var computeUnits C.cl_uint
	C.clGetDeviceInfo(device, C.CL_DEVICE_MAX_COMPUTE_UNITS, C.size_t(unsafe.Sizeof(computeUnits)),
		unsafe.Pointer(&computeUnits), nil)
	info.ComputeUnits = int(computeUnits)

	// Get max work group size
	var maxWorkSize C.size_t
	C.clGetDeviceInfo(device, C.CL_DEVICE_MAX_WORK_GROUP_SIZE, C.size_t(unsafe.Sizeof(maxWorkSize)),
		unsafe.Pointer(&maxWorkSize), nil)
	info.MaxWorkSize = int(maxWorkSize)

	return info
}

// NewGPUMiner creates a new GPU miner instance
func NewGPUMiner(deviceIndex int, batchSize int) (*GPUMiner, error) {
	miner := &GPUMiner{
		batchSize: batchSize,
	}

	// Get platforms
	var numPlatforms C.cl_uint
	ret := C.clGetPlatformIDs(0, nil, &numPlatforms)
	if ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platform count: %s", C.GoString(C.cl_error_string(ret)))
	}

	platforms := make([]C.cl_platform_id, numPlatforms)
	ret = C.clGetPlatformIDs(numPlatforms, &platforms[0], nil)
	if ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platforms: %s", C.GoString(C.cl_error_string(ret)))
	}

	// Find the GPU at the specified index
	gpuIndex := 0
	var selectedPlatform C.cl_platform_id
	var selectedDevice C.cl_device_id
	found := false

	for _, platform := range platforms {
		var numDevices C.cl_uint
		ret = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if ret != C.CL_SUCCESS || numDevices == 0 {
			continue
		}

		devices := make([]C.cl_device_id, numDevices)
		ret = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, numDevices, &devices[0], nil)
		if ret != C.CL_SUCCESS {
			continue
		}

		for _, device := range devices {
			if gpuIndex == deviceIndex {
				selectedPlatform = platform
				selectedDevice = device
				found = true
				break
			}
			gpuIndex++
		}
		if found {
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("GPU device %d not found", deviceIndex)
	}

	miner.platform = selectedPlatform
	miner.device = selectedDevice

	// Get device info
	info := getDeviceInfo(selectedDevice)
	miner.deviceName = info.Name
	miner.maxWorkSize = info.MaxWorkSize

	// Create context
	var errCode C.cl_int
	miner.context = C.clCreateContext(nil, 1, &miner.device, nil, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to create context: %s", C.GoString(C.cl_error_string(errCode)))
	}

	// Create command queue (use deprecated API for OpenCL 1.2 compatibility)
	miner.queue = C.clCreateCommandQueue(miner.context, miner.device, 0, &errCode)
	if errCode != C.CL_SUCCESS {
		C.clReleaseContext(miner.context)
		return nil, fmt.Errorf("failed to create command queue: %s", C.GoString(C.cl_error_string(errCode)))
	}

	// Load and compile kernel
	kernelSource, err := kernelFS.ReadFile("kernel/keccak256.cl")
	if err != nil {
		miner.Close()
		return nil, fmt.Errorf("failed to read kernel source: %v", err)
	}

	kernelSourcePtr := C.CString(string(kernelSource))
	defer C.free(unsafe.Pointer(kernelSourcePtr))

	kernelLen := C.size_t(len(kernelSource))
	miner.program = C.clCreateProgramWithSource(miner.context, 1, &kernelSourcePtr, &kernelLen, &errCode)
	if errCode != C.CL_SUCCESS {
		miner.Close()
		return nil, fmt.Errorf("failed to create program: %s", C.GoString(C.cl_error_string(errCode)))
	}

	// Build program
	ret = C.clBuildProgram(miner.program, 1, &miner.device, nil, nil, nil)
	if ret != C.CL_SUCCESS {
		// Get build log
		var logSize C.size_t
		C.clGetProgramBuildInfo(miner.program, miner.device, C.CL_PROGRAM_BUILD_LOG, 0, nil, &logSize)
		logBuf := make([]byte, logSize)
		C.clGetProgramBuildInfo(miner.program, miner.device, C.CL_PROGRAM_BUILD_LOG, logSize,
			unsafe.Pointer(&logBuf[0]), nil)
		miner.Close()
		return nil, fmt.Errorf("failed to build program: %s\nBuild log:\n%s",
			C.GoString(C.cl_error_string(ret)), string(logBuf))
	}

	// Create kernel
	kernelName := C.CString("mine_create2")
	defer C.free(unsafe.Pointer(kernelName))
	miner.kernel = C.clCreateKernel(miner.program, kernelName, &errCode)
	if errCode != C.CL_SUCCESS {
		miner.Close()
		return nil, fmt.Errorf("failed to create kernel: %s", C.GoString(C.cl_error_string(errCode)))
	}

	return miner, nil
}

// Close releases all OpenCL resources
func (m *GPUMiner) Close() {
	if m.kernel != nil {
		C.clReleaseKernel(m.kernel)
	}
	if m.program != nil {
		C.clReleaseProgram(m.program)
	}
	if m.queue != nil {
		C.clReleaseCommandQueue(m.queue)
	}
	if m.context != nil {
		C.clReleaseContext(m.context)
	}
}

// DeviceName returns the name of the GPU device
func (m *GPUMiner) DeviceName() string {
	return m.deviceName
}

// BatchSize returns the current batch size
func (m *GPUMiner) BatchSize() int {
	return m.batchSize
}

// Mine runs the GPU mining operation
// dataTemplate: 85-byte CREATE2 data with salt prefix already set (bytes 21-40)
// pattern: The address pattern to match
// Returns the result and elapsed time, or nil if nothing found in this batch
func (m *GPUMiner) Mine(dataTemplate []byte, pattern []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	startTime := time.Now()

	var errCode C.cl_int
	batchSize := C.size_t(m.batchSize)

	// Create buffers
	dataTemplateBuf := C.clCreateBuffer(m.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		85, unsafe.Pointer(&dataTemplate[0]), &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create data template buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(dataTemplateBuf)

	patternBuf := C.clCreateBuffer(m.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(len(pattern)), unsafe.Pointer(&pattern[0]), &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create pattern buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(patternBuf)

	// Output buffers
	resultSalt := make([]byte, 12)
	resultSaltBuf := C.clCreateBuffer(m.context, C.CL_MEM_WRITE_ONLY, 12, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create result salt buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(resultSaltBuf)

	resultAddress := make([]byte, 20)
	resultAddressBuf := C.clCreateBuffer(m.context, C.CL_MEM_WRITE_ONLY, 20, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create result address buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(resultAddressBuf)

	found := int32(0)
	foundBuf := C.clCreateBuffer(m.context, C.CL_MEM_READ_WRITE|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(unsafe.Sizeof(found)), unsafe.Pointer(&found), &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create found buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(foundBuf)

	// Set kernel arguments
	patternLen := C.int(len(pattern))
	startNonceC := C.ulong(startNonce)

	C.clSetKernelArg(m.kernel, 0, C.size_t(unsafe.Sizeof(dataTemplateBuf)), unsafe.Pointer(&dataTemplateBuf))
	C.clSetKernelArg(m.kernel, 1, C.size_t(unsafe.Sizeof(patternBuf)), unsafe.Pointer(&patternBuf))
	C.clSetKernelArg(m.kernel, 2, C.size_t(unsafe.Sizeof(patternLen)), unsafe.Pointer(&patternLen))
	C.clSetKernelArg(m.kernel, 3, C.size_t(unsafe.Sizeof(startNonceC)), unsafe.Pointer(&startNonceC))
	C.clSetKernelArg(m.kernel, 4, C.size_t(unsafe.Sizeof(resultSaltBuf)), unsafe.Pointer(&resultSaltBuf))
	C.clSetKernelArg(m.kernel, 5, C.size_t(unsafe.Sizeof(resultAddressBuf)), unsafe.Pointer(&resultAddressBuf))
	C.clSetKernelArg(m.kernel, 6, C.size_t(unsafe.Sizeof(foundBuf)), unsafe.Pointer(&foundBuf))

	// Execute kernel
	localSize := C.size_t(256) // Optimal for Radeon Pro 555X
	globalSize := batchSize
	// Round up to multiple of local size
	if globalSize%localSize != 0 {
		globalSize = ((globalSize / localSize) + 1) * localSize
	}

	ret := C.clEnqueueNDRangeKernel(m.queue, m.kernel, 1, nil, &globalSize, &localSize, 0, nil, nil)
	if ret != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to execute kernel: %s", C.GoString(C.cl_error_string(ret)))
	}

	// Wait for completion
	C.clFinish(m.queue)

	// Read results
	C.clEnqueueReadBuffer(m.queue, foundBuf, C.CL_TRUE, 0, C.size_t(unsafe.Sizeof(found)),
		unsafe.Pointer(&found), 0, nil, nil)

	elapsed := time.Since(startTime)

	if found != 0 {
		// Read the result
		C.clEnqueueReadBuffer(m.queue, resultSaltBuf, C.CL_TRUE, 0, 12,
			unsafe.Pointer(&resultSalt[0]), 0, nil, nil)
		C.clEnqueueReadBuffer(m.queue, resultAddressBuf, C.CL_TRUE, 0, 20,
			unsafe.Pointer(&resultAddress[0]), 0, nil, nil)

		result := &GPUResult{}
		copy(result.SaltSuffix[:], resultSalt)
		copy(result.Address[:], resultAddress)
		result.Nonce = startNonce // Approximate - actual nonce is encoded in salt suffix
		return result, elapsed, nil
	}

	return nil, elapsed, nil
}
