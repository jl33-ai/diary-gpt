import torch


# Check CUDA availability and set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("PyTorch can use your GPU at", device)
    
    # Additional details about the GPU
    print("GPU Details:")
    print(f"Name: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
    
else:
    device = torch.device('cpu')
    print("PyTorch is using your CPU")

# Create a tensor and move it to the selected device
x = torch.tensor([10.0])
x = x.to(device)
print(f'Tensor device: {x.device}')

