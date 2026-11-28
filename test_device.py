from perception.device import init, device, is_cuda, is_fp16, free_vram_gb, mark_resident, is_cpu, device_context

dev, fp16 = init()
print(f"Device: {dev}, FP16: {fp16}")
print(f"is_cuda: {is_cuda()}")
print(f"is_cpu: {is_cpu()}")
print(f"free_vram: {free_vram_gb():.2f} GB")

# Test mark_resident
mark_resident(enable=True)
print(f"After marking resident FP16: {is_fp16()}")

mark_resident(enable=False)
print(f"After unmarking FP16: {is_fp16()}")

# Test device context
with device_context("cuda" if is_cuda() else "cpu"):
    print(f"In context device: {device()}, FP16: {is_fp16()}")

print(f"Back to device: {device()}, FP16: {is_fp16()}")
print("All tests passed!")