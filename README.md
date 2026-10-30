# Agar 1.0 · Native Multimodal Models

A local browser studio built on [BAAI Emu3.5](https://github.com/baaivision/Emu3.5).
The complete upstream source and Apache-2.0 license are retained. Upstream snapshot:
`8fcf1da48f9c5252fe0a0b8dc842e07b6efcd745`. See [the original README](README.upstream.md)
for model details, citations, examples and full backend instructions. Agar additions are
`local_app.py`, `local_config.json`, `configs/local_app_config.py`, tests and this guide;
`model_runtime.py` also gains unique session names and trailing-text persistence.

## Setup

Use Linux (or WSL2 with NVIDIA GPU support), Python 3.12, and a CUDA-compatible
PyTorch environment. These are large models; CPU-only use is not a practical default.
GPU capacity depends on checkpoint, precision and context length; this app does not
claim a tested minimum VRAM. Allow ample disk space for both model and vision-tokenizer
weights. Hub IDs download weights on first use; local paths avoid that download.

```bash
git clone https://github.com/TveJason/Agar1.0-Native-Multimodal-Models.git
cd Agar1.0-Native-Multimodal-Models
python restore_assets.py
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/transformers.txt
python -m pip install flash_attn==2.8.3 --no-build-isolation
python local_app.py
```

Open http://127.0.0.1:7860. Edit `local_config.json` to set `model_path`,
`tokenizer_path`, `vq_path`, `output_dir` and devices. Paths are relative to the
repository root. Use `BAAI/Emu3.5` for both modes; `BAAI/Emu3.5-Image` is optimized
for image generation and should not be used for interleaved generation.

```bash
python local_app.py --config local_config.private.json --port 7861
```

The UI loads the checkpoint lazily once. Choose **Text to image** or
**Interleaved text + images**, enter a prompt, optionally upload a reference image,
and press Generate. A reference image selects upstream X2I for image mode.
Interleaved mode uses upstream visual-guidance templates. Each request is independent.
Text streams in chunks; images appear after a complete image is decoded (no progressive
pixel preview). Stop is cooperative and may wait for the next model event.
Generation is serialized because the runtime holds mutable prompt state.
The service binds to loopback and does not create a public share link.

Every request saves to `outputs/agar/session_<timestamp>_<unique-id>/` by default:
prompt, reference image, generated PNGs/text and `response.json` preserving result order.
The file list allows downloading session files. Restart after changing configuration.

## Existing terminal inference

The upstream scripts remain available:

```bash
# Shared paths from local_config.json; edit prompts in the config for batch requests
python inference.py --cfg configs/local_app_config.py
# Task-specific original presets (edit their model paths/prompts independently)
python inference.py --cfg configs/example_config_t2i.py
python inference.py --cfg configs/example_config_visual_guidance.py
```

## Optional faster offline vLLM backend

Use a **separate** Python 3.12 environment. Upstream pins vLLM 0.11.0 and recommends
CUDA 12.8; its patches are version-specific. Do not upgrade vLLM without checking them.
This path runs offline batch inference, separate from the streaming Transformers UI.

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r requirements/vllm.txt
python -m pip install flash_attn==2.8.3 --no-build-isolation
python src/patch/apply.py
python inference_vllm.py --cfg configs/local_app_config.py --tensor-parallel-size 2 --gpu-memory-utilization 0.7
python src/utils/vis_proto.py --input outputs/agar/proto --output outputs/agar/rendered
```

For one GPU use `--tensor-parallel-size 1` if the checkpoint fits. Batch output is
protobuf; the visualization command exports readable text and images. Edit `prompts`
and task settings in a config before batch inference. Faster performance is an upstream
claim and has not been benchmarked for this wrapper.

## Validation and troubleshooting

```bash
python -m unittest discover -s tests -v
python -m compileall -q local_app.py model_runtime.py configs/local_app_config.py
```

Tests use a fake event runtime to check output ordering, reference-image task selection,
persistence and lock cleanup. They do not load weights or validate GPU generation.
If CUDA runs out of memory, close other GPU processes, use more suitable GPUs or lower
`max_new_tokens`/`image_area` in upstream task presets. Installation/build failures for
FlashAttention usually require matching CUDA, compiler and PyTorch versions. For offline
operation, pre-download checkpoints and configure local paths. Keep the upstream model
licenses/usage conditions when downloading and using weights.

The large upstream `cangerjinkai.ttf` font is stored losslessly as `.ttf.gz` to fit
the upload connector limit. `python restore_assets.py` restores the original bytes
for optional video visualization. All other upstream files retain their original paths.
