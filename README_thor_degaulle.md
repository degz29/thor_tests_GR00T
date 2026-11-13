# Thor session: GR00T N1.7, NVIDIA's pipeline (mixed NVFP4 + BF16), all in ~/Workspace/degaulle

Goal: measure NVIDIA's own deployment on Jetson AGX Thor.
1. Latency per phase (reference: NVIDIA chart, mixed NVFP4 ~40 ms e2e, TRT BF16 ~81 ms; backbone/action head NVFP4 13.6/17.2 ms)
2. Energy per inference (tegrastats)
3. Per-layer breakdown (layer profiler patch). KEY CHECK: in the optimized DiT engine, do the cross-attention
   K/V projection and the adaLN path still run at every DiT call?
4. Real DRAM bandwidth (bw_test.py) vs 273 GB/s spec
5. nsys trace of one NVFP4 run (optional)

Priority if time is short: setup -> NVFP4 build -> NVFP4 latency+power -> bandwidth -> layer profile -> BF16 build -> BF16 latency+power -> nsys.
Total from scratch: ~1-1.5 h setup + ~1 h measurements.

Source of setup commands: https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/
(copy the full `-e PATH=...` value of step 7 from that page; it is a placeholder below)

Legend: [HOST] = Thor shell outside Docker. [CONT] = inside the gr00t-thor container.

---------------------------------------------------------------------------------------------------
## 0. Every new HOST terminal: run these two lines first
```
export D=~/Workspace/degaulle
export HF_HOME=$D/hf_cache
```

## 1. Checks (HOST, 5 min)
```
mkdir -p $D/thor_results $D/hf_cache && cd $D
cat /etc/nv_tegra_release            # R39 (JetPack 7.2)
nvidia-smi                           # CUDA 13.x, NVIDIA Thor
docker --version                     # 28.x+
df -h $D                             # need ~60 GB free (image ~26 GB, ~13 GB per engine build, 7 GB data)
docker images | grep gr00t           # gr00t-thor already built? then skip 4
git lfs version || (sudo apt-get update && sudo apt-get install -y git-lfs)
sudo nvpmodel -q | tee $D/thor_results/nvpmodel_before.txt     # shared machine: restore this at the end
```
Before starting: Hugging Face license for `nvidia/Cosmos-Reason2-2B` accepted, HF token + GitHub token ready.

## 2. Clone both repos (HOST, terminal 1)
```
cd $D
git config --global credential.helper 'cache --timeout=28800'      # GitHub token kept in memory 8 h
git clone https://github.com/degz29/thor_tests_GR00T.git thor_friday   # user degz29, password = token
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout 9c7e746b2cd37a810070a98ef41d290a07e806c2
git lfs install && git lfs pull
file scripts/deployment/thor/wheels/torchcodec-*.whl      # must say "Zip archive", not "ASCII text"
wget -qO- https://www.jetson-ai-lab.com/code-samples/groot_n17_on_thor/download.sh | bash
git status --short | head                                 # files in scripts/deployment modified = patch applied
```
NEVER push from Isaac-GR00T (NVIDIA's repo). Push only from thor_friday.

## 3. Hugging Face: checkpoint (6.5 GB) + calibration dataset (614 MB) (HOST, terminal 1)
```
cd $D/Isaac-GR00T
hf auth login --force && hf auth whoami
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

## 4. Docker image (HOST, terminal 2, in parallel with 3; 15-25 min) - skip if gr00t-thor exists
```
cd $D/Isaac-GR00T/docker && bash build.sh --profile=thor
docker run --rm --runtime nvidia --gpus all gr00t-thor python -c "
import torch, tensorrt as trt
print('torch', torch.__version__); print('TensorRT', trt.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
Expected: torch 2.10.0, TensorRT 10.15.1.29, NVIDIA Thor, (11, 0).
(The image lives in Docker's own storage, not in $D. Normal.)

## 5. Max performance (HOST)
```
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show | tee $D/thor_results/clocks.txt
```

## 6. Start the container (HOST -> becomes CONT)
```
cd $D/Isaac-GR00T
docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v "$D/Isaac-GR00T":/workspace/repo \
  -v "$D/thor_results":/workspace/results \
  -v "$D/thor_friday":/workspace/thor_friday \
  -v "$D/hf_cache":/root/.cache/huggingface \
  -w /workspace/repo \
  -e PYTHONPATH=/workspace/repo \
  -e PATH=<COPY FULL VALUE FROM THE TUTORIAL PAGE, STEP 7> \
  gr00t-thor bash
```
Mount paths must be absolute ($D expands to one). Results written to /workspace/results land in $D/thor_results.

## 7. Build mixed NVFP4 engines FIRST (CONT, 10-15 min)
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
ls ./gr00t_trt_optimized_mixed_nvfp4          # find the engines folder; used as ENG below
```
-> Do section 9 for NVFP4 before building BF16.

## 8. Build BF16 engines (CONT, 10-15 min)
```
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --output-dir ./gr00t_trt_baseline 2>&1 | tee /workspace/results/build_bf16.log
```
Optional if >1 h left: FP8 = section 7 with `--quantization fp8 --output-dir ./gr00t_trt_optimized_fp8`.

---------------------------------------------------------------------------------------------------
## 9. Measurements, per engine set
Set in CONT:
```
ENG=./gr00t_trt_optimized_mixed_nvfp4/engines ; TAG=nvfp4     # for BF16: ENG=./gr00t_trt_baseline/engines ; TAG=bf16
ls $ENG                                                        # check the folder exists
```

### 9A. Latency + power (no profiler!)
HOST, once, nothing running (30 s idle, then Ctrl-C):
```
sudo tegrastats --interval 100 --logfile $D/thor_results/power_idle.log
```
HOST, start logging in the background:
```
sudo tegrastats --start --interval 100 --logfile $D/thor_results/power_nvfp4.log    # power_bf16.log for BF16
```
CONT:
```
date '+%T start' | tee -a /workspace/results/timestamps_$TAG.txt
python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path $ENG \
  --trt-mode n17_full_pipeline --skip-compile --num-iterations 500 \
  2>&1 | tee /workspace/results/bench_$TAG.log
date '+%T end' | tee -a /workspace/results/timestamps_$TAG.txt
```
HOST:
```
sudo tegrastats --stop
head -2 $D/thor_results/power_nvfp4.log      # note rail names: which = GPU, CPU/SoC, total input (VIN)
```
Energy per inference ~ (mean power during steady run - idle power) x latency. Report total and dynamic.
Note: 100 ms samples >> 1 inference, so this is average power over many inferences, not per phase.
Also note how many cameras / the image size the benchmark uses (NVIDIA's chart is 1 camera).

### 9B. Per-layer profile (profiler ON: use the % breakdown, not absolute times)
CONT:
```
cp scripts/deployment/trt_torch.py scripts/deployment/trt_torch.py.orig
patch --dry-run scripts/deployment/trt_torch.py < /workspace/thor_friday/layer_profile.patch
# clean -> patch scripts/deployment/trt_torch.py < /workspace/thor_friday/layer_profile.patch
# not clean -> add the 5 edits by hand, reference: /workspace/thor_friday/trt_torch_patched.py
LAYER_PROFILE=1 python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path $ENG \
  --trt-mode n17_full_pipeline --skip-compile \
  > /workspace/results/layerprof_$TAG.log 2>&1; echo "exit: $?"
grep -n -A14 "^===" /workspace/results/layerprof_$TAG.log | head -150
cp scripts/deployment/trt_torch.py.orig scripts/deployment/trt_torch.py      # restore NVIDIA's file
```
KEY CHECK in the DiT table: still a big fused to_k/to_v MatMul (cross-attention) and a big fused
elementwise block (adaLN) in EVERY call? Note the full layer names.
Reference from A100 (BF16): /workspace/thor_friday/bench_layerprof_a100_bf16.log

### 9C. Bandwidth test (once, CONT)
```
python /workspace/thor_friday/bw_test.py 2>&1 | tee /workspace/results/bw_test.log
```
(A100 reference: 1666.7 GB/s = 82% of spec. Thor spec 273 GB/s.)

### 9D. nsys trace (NVFP4 only, if time, CONT)
```
which nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null
nsys profile -o /workspace/results/nvfp4_trace --trace=cuda,nvtx --force-overwrite true \
  python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_trt_optimized_mixed_nvfp4/engines \
  --trt-mode n17_full_pipeline --skip-compile --num-iterations 20
nsys stats --report cuda_gpu_kern_sum /workspace/results/nvfp4_trace.nsys-rep > /workspace/results/nvfp4_kernels.txt
```

---------------------------------------------------------------------------------------------------
## 10. Save + clean up (HOST)
```
cd $D/thor_friday && mkdir -p results && cp -r $D/thor_results/* results/
git add results && git commit -m "Thor results $(date +%F)" && git push
cat $D/thor_results/nvpmodel_before.txt      # then: sudo nvpmodel -m <that mode number>
```
Backup: zip $D/thor_results and send it to yourself.

## Troubleshooting
- torchcodec wheel is "ASCII text" -> `git lfs pull` did not run.
- Build fails on Cosmos-Reason2 -> license not accepted / `hf auth login` missing (HF_HOME must be set as in 0).
- Empty /workspace/results or /workspace/thor_friday in CONT -> a -v path was not absolute / $D not set in that terminal.
- git asks no password and fails -> `git clone https://degz29@github.com/degz29/thor_tests_GR00T.git thor_friday`
- Thor cannot reach GitHub -> clone on Chao's PC, then `scp -r thor_tests_GR00T <user>@<thor-ip>:$D/thor_friday`

## Checklist before leaving Thor
- [ ] clocks.txt, nvpmodel_before.txt, build_nvfp4.log, build_bf16.log
- [ ] bench_nvfp4.log + power_nvfp4.log + power_idle.log + timestamps_nvfp4.txt
- [ ] layerprof_nvfp4.log (+ key check answered)
- [ ] bw_test.log
- [ ] bench_bf16.log + power_bf16.log + timestamps_bf16.txt + layerprof_bf16.log
- [ ] nvfp4_trace / nvfp4_kernels.txt (optional)
- [ ] trt_torch.py restored, power mode restored
- [ ] everything pushed / mailed
