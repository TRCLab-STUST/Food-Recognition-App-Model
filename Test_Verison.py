import torch

# 檢查CUDA是否可用
cuda_available = torch.cuda.is_available()
print("CUDA 可用:", cuda_available)

# 檢查CUDA版本
if cuda_available:
    cuda_version = torch.version.cuda
    print("CUDA 版本:", cuda_version)

# 檢查cuDNN是否可用
if cuda_available:
    cudnn_available = torch.backends.cudnn.enabled
    print("cuDNN 可用:", cudnn_available)

# 檢查cuDNN版本
if cuda_available and cudnn_available:
    cudnn_version = torch.backends.cudnn.version()
    print("cuDNN 版本:", cudnn_version)
