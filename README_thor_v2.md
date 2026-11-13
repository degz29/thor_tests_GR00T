# Thor session v2: GR00T N1.7, NVIDIA's pipeline (mixed NVFP4 + BF16)
Everything in /home/cfang/Workspace/degaulle. Includes today's fixes (sudo docker, own hf venv, full paths).

Goal: NVIDIA's own deployment on Thor -> latency, power, layer profile, real bandwidth.
Priority if short on time: NVFP4 build -> NVFP4 latency (+power) -> bandwidth -> layer profile -> BF16 -> nsys.

[HOST] = Thor shell, outside Docker.  [CONT] = inside the gr00t-thor container.
Paste in a terminal = Ctrl+Shift+V.

====================================================================================================
## 0. EVERY new HOST terminal: paste this first (full paths, NO ~)
```
export HOME=/home/cfang
export D=/home/cfang/Workspace/degaulle
export HF_HOME=$D/hf_cache
export HF=$D/hfenv/bin/hf
cd $D && pwd
```
`pwd` must print /home/cfang/Workspace/degaulle. If any path you see contains a folder named `~`, stop: D was set wrong.
Do NOT activate any venv / conda. Never type `rm -rf ~`.

====================================================================================================
## 1. Status check (HOST) - skip every step that is already done
```
ls $D                                                        # Isaac-GR00T thor_friday thor_results hf_cache hfenv
ls $D/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10    # checkpoint there?
ls $D/Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta   # dataset + modality.json?
$HF --version                                                # hf tool there?
sudo docker images | grep gr00t || echo "IMAGE NOT BUILT"
df -h $D                                                     # need ~60 GB free
```

====================================================================================================
## 2. Folders + repos (HOST)  [skip what exists]
```
mkdir -p $D/thor_results $D/hf_cache
cd $D
git config --global credential.helper 'cache --timeout=28800'
git clone https://github.com/degz29/thor_tests_GR00T.git thor_friday     # user degz29, password = GitHub token
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd $D/Isaac-GR00T
git checkout 9c7e746b2cd37a810070a98ef41d290a07e806c2
git lfs install && git lfs pull
file scripts/deployment/thor/wheels/torchcodec-*.whl       # must say "Zip archive"
wget -qO- https://www.jetson-ai-lab.com/code-samples/groot_n17_on_thor/download.sh | bash
```
Never push from Isaac-GR00T. Push only from thor_friday.

====================================================================================================
## 3. Docker image (HOST, TERMINAL B - start it now, runs 15-25 min)  [skip if built]
```
(paste section 0 first)
cd $D/Isaac-GR00T/docker && sudo bash build.sh --profile=thor
```
Test when done:
```
sudo docker run --rm --runtime nvidia --gpus all gr00t-thor python -c "import torch, tensorrt as trt; print(torch.__version__, trt.__version__, torch.cuda.get_device_name(0))"
```
Expected: 2.10.0  10.15.1.29  NVIDIA Thor.  Docker always needs `sudo` on this machine.

====================================================================================================
## 4. hf tool + downloads (HOST, TERMINAL A, in parallel with 3)
hf tool (skip if `$HF --version` works):
```
/usr/bin/python3 -m venv $D/hfenv
$D/hfenv/bin/pip install -U "huggingface_hub[cli]"
$HF --version
```
Login (HF token: huggingface.co -> Settings -> Access Tokens -> Read. License of nvidia/Cosmos-Reason2-2B accepted):
```
$HF auth login --force          # paste token, answer n to git credential
$HF auth whoami
```
Downloads (~7 GB):
```
cd $D/Isaac-GR00T && pwd        # /home/cfang/Workspace/degaulle/Isaac-GR00T
$HF download nvidia/GR00T-N1.7-LIBERO --include "libero_10/*" --local-dir checkpoints/GR00T-N1.7-LIBERO
$HF download --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/
cp examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
ls checkpoints/GR00T-N1.7-LIBERO/libero_10
```

====================================================================================================
## 5. Max performance (HOST)
```
sudo nvpmodel -q | tee $D/thor_results/nvpmodel_before.txt    # restore at the end
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show > $D/thor_results/clocks.txt
```

====================================================================================================
## 6. Start the container (HOST -> CONT)   needs: image + checkpoint + dataset
Copy the full PATH value from https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/ step 7.
```
cd $D/Isaac-GR00T
sudo docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v "$D/Isaac-GR00T":/workspace/repo \
  -v "$D/thor_results":/workspace/results \
  -v "$D/thor_friday":/workspace/thor_friday \
  -v "$D/hf_cache":/root/.cache/huggingface \
  -w /workspace/repo -e PYTHONPATH=/workspace/repo \
  -e PATH=<FULL VALUE FROM TUTORIAL STEP 7> \
  gr00t-thor bash
```
Check inside: `ls /workspace/results /workspace/thor_friday checkpoints/GR00T-N1.7-LIBERO/libero_10` (none empty).

====================================================================================================
## 7. Build mixed NVFP4 engines FIRST (CONT, 10-15 min)
```
uv pip install "nvidia-modelopt[onnx]==0.39.0"
python gr00t/data/stats.py --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot --embodiment-tag LIBERO_PANDA
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
  --output-dir ./gr00t_trt_optimized_mixed_nvfp4 \
  --execution-profile optimized --quantization mixed_nvfp4 \
  --calib-dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot --calib-size 10 \
  2>&1 | tee /workspace/results/build_nvfp4.log
ls ./gr00t_trt_optimized_mixed_nvfp4
```
-> measure NVFP4 (section 9) before building BF16.

## 8. Build BF16 engines (CONT, 10-15 min)
```
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
  --output-dir ./gr00t_trt_baseline 2>&1 | tee /workspace/results/build_bf16.log
```

====================================================================================================
## 9. Measurements (per engine set)
CONT:
```
ENG=./gr00t_trt_optimized_mixed_nvfp4/engines; TAG=nvfp4      # BF16: ENG=./gr00t_trt_baseline/engines; TAG=bf16
ls $ENG
```

### 9A. Latency + power (profiler OFF)
HOST (once, nothing running, 30 s, Ctrl-C):
```
sudo tegrastats --interval 100 --logfile $D/thor_results/power_idle.log
```
HOST (start background log; use power_bf16.log for BF16):
```
sudo tegrastats --start --interval 100 --logfile $D/thor_results/power_nvfp4.log
```
CONT:
```
date '+%T start' | tee -a /workspace/results/timestamps_$TAG.txt
python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 --trt-engine-path $ENG \
  --trt-mode n17_full_pipeline --skip-compile --num-iterations 500 \
  2>&1 | tee /workspace/results/bench_$TAG.log
date '+%T end' | tee -a /workspace/results/timestamps_$TAG.txt
```
HOST:
```
sudo tegrastats --stop
head -2 $D/thor_results/power_nvfp4.log      # note rail names (GPU, CPU/SoC, VIN total)
```
Energy/inference ~ (mean power during run - idle) x latency. Also note number of cameras / image size used.

### 9B. Layer profile (profiler ON: use % shares only)
CONT:
```
cp scripts/deployment/trt_torch.py scripts/deployment/trt_torch.py.orig
patch --dry-run scripts/deployment/trt_torch.py < /workspace/thor_friday/layer_profile.patch
patch scripts/deployment/trt_torch.py < /workspace/thor_friday/layer_profile.patch     # only if dry-run clean
LAYER_PROFILE=1 python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 --trt-engine-path $ENG \
  --trt-mode n17_full_pipeline --skip-compile \
  > /workspace/results/layerprof_$TAG.log 2>&1; echo "exit: $?"
grep -n -A14 "^===" /workspace/results/layerprof_$TAG.log | head -150
cp scripts/deployment/trt_torch.py.orig scripts/deployment/trt_torch.py
```
If dry-run is not clean: edit by hand using /workspace/thor_friday/trt_torch_patched.py.
KEY CHECK (DiT): fused to_k/to_v MatMul and big adaLN elementwise block still in EVERY call?
A100 reference: /workspace/thor_friday/bench_layerprof_a100_bf16.log

### 9C. Bandwidth (once, CONT)
```
python /workspace/thor_friday/bw_test.py 2>&1 | tee /workspace/results/bw_test.log
```

### 9D. nsys (optional, NVFP4, CONT)
```
nsys profile -o /workspace/results/nvfp4_trace --trace=cuda,nvtx --force-overwrite true \
  python scripts/deployment/benchmark_inference.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_trt_optimized_mixed_nvfp4/engines --trt-mode n17_full_pipeline --skip-compile --num-iterations 20
nsys stats --report cuda_gpu_kern_sum /workspace/results/nvfp4_trace.nsys-rep > /workspace/results/nvfp4_kernels.txt
```

====================================================================================================
## 10. Save + clean up (HOST)
```
cd $D/thor_friday && mkdir -p results && cp -r $D/thor_results/* results/
git add results && git commit -m "Thor results $(date +%F)" && git push
cat $D/thor_results/nvpmodel_before.txt          # sudo nvpmodel -m <that number>
$HF auth logout; git credential-cache exit       # shared machine: drop tokens
```

====================================================================================================
## Troubleshooting (all seen today)
| Symptom | Fix |
|---|---|
| `permission denied ... docker API` | put `sudo` before every docker command |
| `no such file ... Isaac-GR00T/docker` | clone first (section 2); check `echo $D` |
| `hf: command not found` | section 4 venv; always call `$HF` (full path) |
| `user site packages are not visible in this venv` | don't use --user; use `$D/hfenv` (section 4) |
| `could not determine home directory` | `export HOME=/home/cfang` (no `$` before HOME) |
| path with a folder named `~` | D set with literal ~ -> paste section 0 again (full paths) |
| python errors pointing to an old hfenv | close terminal, open a new one, paste section 0, activate nothing |
| torchcodec wheel = "ASCII text" | `git lfs pull` |
| build fails on Cosmos-Reason2 | accept license on HF, `$HF auth login` |
| empty /workspace/... in container | `$D` not set in the terminal that ran docker run |
| keyboard types Chinese | Shift, or Super+Space, or Ctrl+Space; top bar -> English (US) |

## Checklist
- [ ] clocks.txt, nvpmodel_before.txt, build_nvfp4.log
- [ ] bench_nvfp4.log, power_nvfp4.log, power_idle.log, timestamps_nvfp4.txt
- [ ] bw_test.log
- [ ] layerprof_nvfp4.log (+ key check)
- [ ] BF16: build_bf16.log, bench_bf16.log, power_bf16.log, layerprof_bf16.log
- [ ] trt_torch.py restored, power mode restored, tokens logged out
- [ ] pushed / mailed
