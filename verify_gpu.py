import torch

if torch.cuda.is_available():
    print("✅ Success! PyTorch can see your GPU.")
    gpu_count = torch.cuda.device_count()
    print(f"   - Found {gpu_count} GPU(s).")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"   - GPU Name: {gpu_name}")
else:
    print("❌ Error: PyTorch cannot detect your GPU.")
    print("   - Please ensure your NVIDIA drivers and CUDA-enabled PyTorch are installed correctly.")