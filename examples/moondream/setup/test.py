import safetensors.torch
import torch

sd = safetensors.torch.load_file("moondream2.safetensors")
w = sd["model.vision.patch_emb.weight"]              # should be shape [1152,588]
print("py shape:", w.shape)
print("py first row:", w[0,:10].tolist())