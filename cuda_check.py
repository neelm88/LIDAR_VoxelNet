import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Iterate through available GPUs and print their properties
    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print("CUDA is not available. Using CPU.")