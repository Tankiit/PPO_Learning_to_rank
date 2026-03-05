#!/usr/bin/env python3
"""Test script to verify device detection and MPS support"""

import torch

print("="*60)
print("DEVICE DETECTION TEST")
print("="*60)

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

print(f"\nMPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print("MPS backend is built and available")
    print("Device will use: MPS (Metal Performance Shaders)")

print(f"\nPyTorch version: {torch.__version__}")

# Test tensor creation on each device
print("\n" + "="*60)
print("DEVICE TESTS")
print("="*60)

# CPU (always works)
x = torch.randn(3, 3)
print(f"\nCPU tensor created: {x.device}")

# CUDA if available
if torch.cuda.is_available():
    x_cuda = x.cuda()
    print(f"CUDA tensor created: {x_cuda.device}")

# MPS if available
if torch.backends.mps.is_available():
    x_mps = x.to('mps')
    print(f"MPS tensor created: {x_mps.device}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if torch.cuda.is_available():
    print("\n✓ Use: python train_ranking_model.py --use_cuda")
elif torch.backends.mps.is_available():
    print("\n✓ Use: python train_ranking_model.py")
    print("  (MPS will be used automatically, no --use_cuda flag needed)")
    print("  (Recommended: --num_workers 2 or higher for best performance)")
else:
    print("\n✓ Use: python train_ranking_model.py")
    print("  (Will use CPU)")

print()
