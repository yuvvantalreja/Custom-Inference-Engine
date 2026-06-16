import torch
import torch.nn as nn

# 1. Define the 2D Convolutional layer
# Arguments: in_channels=3 (RGB), out_channels=16, kernel_size=3x3, stride=1, padding=1
conv_layer = nn.Conv2d(
    in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
)

# 2. Create a dummy input tensor matching the required shape:
# Shape: (batch_size, in_channels, height, width)
input_tensor = torch.randn(1, 3, 2, 2, dtype=torch.float32)

x_max, x_min = input_tensor.max() , input_tensor.min()

# Per Tensor Quantization
scale = (x_max - x_min) / 255
zero_point = torch.round(-x_min / scale)

quantized = torch.round(input_tensor / scale) + zero_point
quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)


# 3. Pass the input through the layer
out = conv_layer(input_tensor)
quant_out = conv_layer(quantized)

recovered = scale * (quant_out - zero_point)

print("Input: ", input_tensor)
print("Output: ", out)
print("Recovered Out: ", recovered)

print(f"{scale}")
print(f"{zero_point}")
# Expected output: torch.Size([1, 16, 32, 32])


# import torch

# input_tensor = torch.randn(1, 3, 2, 2)

# x_min = input_tensor.min()
# x_max = input_tensor.max()

# scale = (x_max - x_min) / 255

# zero_point = torch.round(-x_min / scale)

# quantized = torch.round(input_tensor / scale) + zero_point
# quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

# dequantized = scale * (quantized.float() - zero_point)

# print("Original:")
# print(input_tensor)

# print("\nScale:", scale.item())
# print("Zero Point:", zero_point.item())

# print("\nQuantized:")
# print(quantized)

# print("\nDequantized:")
# print(dequantized)
