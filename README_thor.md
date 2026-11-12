# Thor session: GR00T N1.7, NVIDIA's pipeline (mixed NVFP4 + BF16)

Goal: measure NVIDIA's own deployment on Jetson AGX Thor.
1. Latency per phase (compare: backbone/action head NVFP4 13.6/17.2 ms, BF16 27.0/45.0 ms, e2e ~40 / ~81 ms)
2. Energy per inference (tegrastats)
3. Per-layer breakdown (layer profiler patch) - KEY CHECK: in the optimized DiT engine, do the
   cross-attention K/V projection and the adaLN path still run at every DiT call?
4. Real achievable DRAM bandwidth (bw_test.py) vs 273 GB/s spec
5. nsys trace of one NVFP4 run (true kernel times, cross-check of 3)

Order if time is short: setup -> NVFP4 build -> NVFP4 measurements (A) -> bandwidth -> BF16 build -> BF16 measurements -> nsys.

Source of all setup commands: https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/
(copy the full `-e PATH=...` line of step 7 from that page; it is truncated below)

---------------------------------------------------------------------------------------------------
## 0. Before anything (host, 5 min)

```
mkdir -p ~/thor_results && cd ~/thor_results
cat /etc/nv_tegra_release            # must show R39 (JetPack 7.2)
nvidia-smi                           # CUDA 13.x, NVIDIA Thor
docker --version                     # 28.x+
git lfs version
df -h ~                              # need ~60 GB free (container 26 GB + ~13 GB per engine build)
docker images | grep gr00t           # is gr00t-thor already there?
ls ~/Isaac-GR00T 2>/dev/null && (cd ~/Isaac-GR00T && git log -1 --oneline)   # existing checkout? which commit?
find ~ -name "*.engine" -path "*gr00t*" 2>/dev/null | head                     # engines already built?
```
If container + engines (mixed NVFP4 and BF16) already exist: skip to section 3.

## 1. Max performance (host)

```
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show | tee ~/thor_results/clocks.txt
```

## 2. Setup from scratch (only if missing, ~1-1.5 h total)

### 2.1 Git LFS, clone, pinned commit, NVIDIA's optimization patch
```
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
cd ~ && git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout 9c7e746b2cd37a810070a98ef41d290a07e806c2
git lfs pull
file scripts/deployment/thor/wheels/torchcodec-*.whl      # must say Zip archive, not ASCII text
wget -qO- https://www.jetson-ai-lab.com/code-samples/groot_n17_on_thor/download.sh | bash
```

### 2.2 Hugging Face (license for nvidia/Cosmos-Reason2-2B accepted beforehand; token ready)
```
hf auth login --force
hf auth whoami
```

### 2.3 Checkpoint (6.5 GB) + calibration dataset (614 MB) -- run in parallel with 2.4
```
hf download nvidia/GR00T-N1.7-LIBERO \
  libero_10/config.json libero_10/embodiment_id.json \
  libero_10/model-00001-of-00002.safetensors \
  libero_10/model-00002-of-00002.safetensors \
  libero_10/model.safetensors.index.json \
  libero_10/processor_config.json libero_10/statistics.json \
  libero_10/experiment_cfg/conf.yaml \
  libero_10/experiment_cfg/config.yaml \
  libero_10/experiment_cfg/dataset_statistics.json \
  --local-dir checkpoints/GR00T-N1.7-LIBERO

hf download --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

cp examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
```

### 2.4 Docker image (15-25 min, pulls ~10 GB base image) -- second terminal
```
cd ~/Isaac-GR00T/docker && bash build.sh --profile=thor && cd ..
docker run --rm --runtime nvidia --gpus all gr00t-thor python -c "
import torch, tensorrt as trt
print('torch', torch.__version__); print('TensorRT', trt.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
Expected: torch 2.10.0, TensorRT 10.15.1.29, NVIDIA Thor, (11, 0).

### 2.5 Launch the container (all following steps run INSIDE it unless marked "host")
Mount the results folder too, so logs land on the host.
```
cd ~/Isaac-GR00T
docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "$PWD":/workspace/repo \
  -v "$HOME/thor_results":/workspace/results \
  -v "$HOME/thor_friday":/workspace/thor_friday \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  -e PYTHONPATH=/workspace/repo \
  -e PATH=<COPY FULL VALUE FROM THE TUTORIAL PAGE, STEP 7> \
  gr00t-thor bash
```

### 2.6 Build mixed NVFP4 engines FIRST (~10-15 min; prints NVIDIA's own benchmark at the end)
```
uv pip install "nvidia-modelopt[onnx]==0.39.0"
python -c "import modelopt; print('modelopt', modelopt.__version__)"
python gr00t/data/stats.py \
  --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot \
  --embodiment-tag LIBERO_PANDA

python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --output-dir ./gr00t_trt_optimized_mixed_nvfp4 \
  --execution-profile optimized \
  --quantization mixed_nvfp4 \
  --calib-dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot \
  --calib-size 10 2>&1 | tee /workspace/results/build_nvfp4.log
```
-> Then do section 3 for NVFP4 before building BF16.

### 2.7 Build BF16 engines (~10-15 min)
```
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --output-dir ./gr00t_trt_baseline 2>&1 | tee /workspace/results/build_bf16.log
```
(FP8 only if >1 h left: same as 2.6 with `--quantization fp8 --output-dir ./gr00t_trt_optimized_fp8`.)

---------------------------------------------------------------------------------------------------
## 3. Measurements (per engine set: NVFP4 = ./gr00t_trt_optimized_mixed_nvfp4/engines,
##    BF16 = ./gr00t_trt_baseline/engines -- check the exact engines folder with ls)

### 3A. Latency + power (no profiler!) -- two terminals
Host terminal (power log, start first, stop with Ctrl-C after the benchmark):
```
sudo tegrastats --interval 100 --logfile ~/thor_results/power_idle.log      # 30 s with nothing running, Ctrl-C
sudo tegrastats --interval 100 --logfile ~/thor_results/power_nvfp4.log     # then start the benchmark
```
Container terminal:
```
date '+%T start' | tee -a /workspace/results/timestamps.txt
python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_trt_optimized_mixed_nvfp4/engines \
  --trt-mode n17_full_pipeline --skip-compile --num-iterations 200 \
  2>&1 | tee /workspace/results/bench_nvfp4.log
date '+%T end' | tee -a /workspace/results/timestamps.txt
```
Power rail names differ per board: look at the first line of the log and note which is GPU / CPU-SoC / total.
Energy per inference ~ (mean running power - idle power) x latency.

### 3B. Per-layer profile (profiler ON; use its % breakdown, not its absolute times)
NVIDIA's download.sh patch modifies scripts/deployment/ (13 files), so first try the patch, else edit by hand:
```
cp scripts/deployment/trt_torch.py scripts/deployment/trt_torch.py.orig
patch --dry-run scripts/deployment/trt_torch.py < /workspace/thor_friday/layer_profile.patch
# if clean: patch scripts/deployment/trt_torch.py < /workspace/thor_friday/layer_profile.patch
# if not: add the 5 edits by hand, reference = /workspace/thor_friday/trt_torch_patched.py
LAYER_PROFILE=1 python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_trt_optimized_mixed_nvfp4/engines \
  --trt-mode n17_full_pipeline --skip-compile \
  > /workspace/results/layerprof_nvfp4.log 2>&1; echo "exit: $?"
grep -n -A14 "^===" /workspace/results/layerprof_nvfp4.log | head -150
```
KEY CHECK in the DiT table: is there still a big fused to_k/to_v MatMul over the cross-attention blocks,
and a big fused elementwise block (adaLN) in EVERY call? Note the full layer names.
Restore NVIDIA's file afterwards: `cp scripts/deployment/trt_torch.py.orig scripts/deployment/trt_torch.py`

### 3C. Bandwidth test (once)
```
python /workspace/thor_friday/bw_test.py 2>&1 | tee /workspace/results/bw_test.log
```

### 3D. nsys trace (NVFP4 only, if time)
```
which nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null   # container, else host
nsys profile -o /workspace/results/nvfp4_trace --trace=cuda,nvtx --force-overwrite true \
  python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_trt_optimized_mixed_nvfp4/engines \
  --trt-mode n17_full_pipeline --skip-compile --num-iterations 20
nsys stats --report cuda_gpu_kern_sum /workspace/results/nvfp4_trace.nsys-rep > /workspace/results/nvfp4_kernels.txt
```

---------------------------------------------------------------------------------------------------
## 4. Save everything (host)
```
cd ~/thor_results && cp ~/Isaac-GR00T/scripts/deployment/trt_torch.py . 2>/dev/null
cd ~/thor_friday && mkdir -p results && cp -r ~/thor_results/* results/
git add results && git commit -m "Thor results $(date +%F)" && git push      # token as password
```
Backup: zip ~/thor_results and mail it to yourself.

## Checklist before leaving Thor
- [ ] clocks.txt, build logs
- [ ] bench_nvfp4.log + power_nvfp4.log + power_idle.log + timestamps.txt
- [ ] layerprof_nvfp4.log (+ key check answered)
- [ ] bw_test.log
- [ ] bench_bf16.log + power_bf16.log + layerprof_bf16.log
- [ ] nvfp4_trace (optional)
- [ ] everything pushed / mailed
