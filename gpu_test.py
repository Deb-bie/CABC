import torch
import time

def test_gpu_cuda():
    """
    Tests if PyTorch is using the GPU and CUDA successfully.
    Prints detailed information about the GPU and performance comparison.
    """

    print("--- PyTorch GPU/CUDA Test Script ---")

    # 1. Check CUDA Availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nCUDA is available! PyTorch is using GPU.")
        print(f"Device name: {torch.cuda.get_device_name(0)}")  # Get GPU name
        print(f"Device count: {torch.cuda.device_count()}")     # Get number of GPUs
        print(f"Current device: {torch.cuda.current_device()}") # Get index of current GPU
        print(f"CUDA version: {torch.version.cuda}")           # Get CUDA version
        print(f"cuDNN version: {torch.backends.cudnn.version()}") # Get cuDNN version

        # Optional: Check for specific CUDA capabilities (if needed for your use case)
        # print(f"Device capability: {torch.cuda.get_device_capability(0)}")

    else:
        device = torch.device('cpu')
        print("\nCUDA is NOT available. PyTorch is using CPU.")
        print("Please ensure you have installed CUDA drivers and a CUDA-enabled GPU.")
        print("Refer to PyTorch installation instructions for CUDA setup.")
        return  # Exit the test early if no CUDA

    print(f"\nUsing device: {device}")

    # 2. Create a Tensor and Move it to the Device
    print("\n--- Tensor Creation and Device Transfer ---")
    tensor_cpu = torch.randn(5, 5)
    print(f"CPU Tensor:\n{tensor_cpu}")
    print(f"Tensor on CPU? {tensor_cpu.is_cuda}")

    tensor_gpu = tensor_cpu.to(device)  # Move tensor to GPU
    print(f"\nGPU Tensor:\n{tensor_gpu}")
    print(f"Tensor on GPU? {tensor_gpu.is_cuda}")

    # 3. Perform a Simple Operation on GPU and CPU and Time it
    print("\n--- Performance Comparison (GPU vs CPU) ---")
    size = 10000
    iterations = 1

    # GPU Performance
    gpu_tensor = torch.randn(size, size).to(device)
    start_time_gpu = time.time()
    for _ in range(iterations):
        gpu_tensor @ gpu_tensor  # Matrix multiplication on GPU
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu

    # CPU Performance
    cpu_tensor = torch.randn(size, size)
    start_time_cpu = time.time()
    for _ in range(iterations):
        cpu_tensor @ cpu_tensor  # Matrix multiplication on CPU
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu

    print(f"\nGPU Time ({iterations} iterations of {size}x{size} matrix multiplication): {gpu_time:.4f} seconds")
    print(f"CPU Time ({iterations} iterations of {size}x{size} matrix multiplication): {cpu_time:.4f} seconds")

    if device.type == 'cuda':
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf') # Avoid division by zero
        print(f"\nSpeedup (CPU/GPU): {speedup:.2f}x")
        if speedup > 1.5: # Heuristic, adjust as needed
            print("\nPerformance improvement suggests GPU is being used effectively!")
        else:
            print("\nPerformance improvement might be minimal. Double-check GPU usage and problem complexity.")
    else:
        print("\nNo performance comparison possible as CUDA is not available.")

    print("\n--- Test Script Completed ---")

if __name__ == "__main__":
    test_gpu_cuda()