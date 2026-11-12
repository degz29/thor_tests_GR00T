import torch

N_BYTES = 1024**3                      # 1 GB per buffer: big enough that caches don't matter
ITERS = 20

src = torch.empty(N_BYTES // 4, dtype=torch.float32, device="cuda")   # 1 GB = 256 M floats x 4 bytes
dst = torch.empty_like(src)

src.fill_(1.0)
dst.fill_(0.0)
#warmup
for _ in range(5):    
    dst.copy_(src)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    dst.copy_(src)
end.record()
torch.cuda.synchronize()
elapsed_time_ms = start.elapsed_time(end)
bw = (2 * N_BYTES * ITERS) / (elapsed_time_ms * 1e-3) / (1e9)  # GB/s
print(f"{torch.cuda.get_device_name(0)}: {bw:.1f} GB/s")
print(f"Bandwidth: {bw:.2f} GB/s")