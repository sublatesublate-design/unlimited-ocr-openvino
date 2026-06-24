# Unlimited-OCR OpenVINO Adaptation Notes

This directory records the adaptation decisions, verification results, and
remaining performance work.

## Current snapshot

- GitHub repo: `baidu/Unlimited-OCR`, shallow cloned into the workspace.
- Hugging Face remote code copied into `research/hf_remote_code/`.
- Model weights are downloaded under `models/Unlimited-OCR/` and ignored by git.
- OpenVINO adaptation package lives in `openvino_adapt/`.

## Findings so far

- The GitHub repo mainly contains README examples, `infer.py`, assets, and a custom SGLang wheel.
- The actual Hugging Face model implementation lives in remote-code files:
  - `modeling_unlimitedocr.py`
  - `modeling_deepseekv2.py`
  - `deepencoder.py`
  - `configuration_deepseek_v2.py`
- `infer.py` sends PDF pages as separate SGLang requests. The one-shot multi-page path is in `UnlimitedOCRForCausalLM.infer_multi()`.
- The model has three practical export regions:
  - image encoder stack: `sam_model`, `vision_model`
  - `projector`: visual features to decoder hidden size
  - decoder: DeepseekV2-style causal LM with custom sliding-window/ring-cache attention
- Base-mode 1024 visual tokens are `[1, 273, 1280]` per page: `16 * (16 + 1) + 1`.

## Why decoder is split

The custom `SlidingWindowLlamaAttention` keeps prefill KV and uses a bounded ring for generated tokens. That means a useful OpenVINO port is not just one static `generate()` export. It likely needs:

- a prefill graph,
- a one-token decode graph,
- host-side KV ownership,
- explicit ring-buffer updates,
- careful attention-mask and position-id handling.

The adapter now implements this split with `decoder_prefill_kv`,
`decoder_decode_one`, and a Python host loop.

## Milestone status

1. Static inspection: done with `python research/inspect_unlimitedocr.py`.
2. Model snapshot: downloaded under `models/Unlimited-OCR/`.
3. Environment check: done with `python -m openvino_adapt.env_check`.
4. Vision export and comparison: done; OpenVINO output matches PyTorch closely.
5. Explicit-KV decoder export: done for the default one-page prompt length.
6. Host R-SWA loop: done with fixed prefill KV plus generated-token ring window.
7. OCR runner: done for image, image directory, and PDF inputs.
8. Remaining milestone: make sparse MoE or quantized decoder fast enough for comfortable long outputs.

## Current adapter commands

```shell
python -m openvino_adapt.env_check
python -m openvino_adapt.cache_design
python -m openvino_adapt.export_openvino --component projector
python -m openvino_adapt.export_openvino --component embed_tokens
python -m openvino_adapt.export_openvino --component vision_tokens
python -m openvino_adapt.export_openvino --component decoder_no_cache
```

All export commands are dry runs unless `--allow-weight-download` is present.

## Verified OpenVINO artifacts

- `projector.xml`: exported and CPU smoke-tested.
- `embed_tokens.xml`: exported and CPU smoke-tested.
- `vision_tokens.xml`: exported, CPU-tested, and numerically compared against PyTorch.
  - PyTorch/OpenVINO shape: `[1, 273, 1280]`
  - max absolute difference observed: about `7.6e-5`
  - mean absolute difference observed: about `5.1e-7`
- `decoder_no_cache.xml` at `seq_len=16`: exported and CPU smoke-tested.
- `decoder_no_cache.xml` at `seq_len=277`: exported and used in the no-cache prefill pipeline.
- `decoder_decode_one.xml` explicit-KV dense-MoE graph:
  - exported under `openvino_models/unlimited_ocr_kv_dense/`
  - CPU smoke-tested with 27 inputs and 25 outputs
- `decoder_prefill_kv.xml` explicit-KV dense-MoE graph:
  - exported under `openvino_models/unlimited_ocr_kv_dense_prefill277/`
  - CPU smoke-tested with 3 inputs and 25 outputs
- OpenVINO explicit-KV graphs were compared against PyTorch explicit wrappers:
  - prefill logits max absolute difference: about `6.9e-5`
  - prefill logits mean absolute difference: about `2.3e-6`
  - decode logits max absolute difference: about `3.2e-5`
  - decode logits mean absolute difference: about `5.3e-6`
- End-to-end OpenVINO greedy generation loop runs with:
  - `python -m openvino_adapt.run_generate_openvino --image path/to/page.png --max-new-tokens 4`
- OCR-style CLI runs with image/image-dir/PDF inputs:
  - `python -m openvino_adapt.run_ocr_openvino --image path/to/page.png --output-dir outputs_openvino --max-new-tokens 128`
  - `python -m openvino_adapt.run_ocr_openvino --pdf path/to/document.pdf --output-dir outputs_openvino --max-new-tokens 128`
- Continuous multi-page artifacts were exported and smoke-tested:
  - `openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml`
  - `openvino_models/unlimited_ocr_kv_dense_past677/decoder_decode_one.xml`
  - prompt: `<image>document parsing.`
  - page count: 2
  - compressed FP16: true
- Smoke-tested outputs:
  - single-image CLI writes `page_0001.md`, `combined.md`, and `manifest.json`
  - one-page PDF CLI writes `page_0001.md`, `combined.md`, and `manifest.json`
  - two-page continuous image-dir CLI writes `continuous.md`, `page_*.md`,
    `combined.md`, and `manifest.json`
- Adapter metadata sidecars are used for shape validation:
  - `decoder_prefill_kv.json` stores `seq_len`
  - `decoder_decode_one.json` stores `past_len`
  - prompt/past-length mismatches fail early with an export command suggestion
- Host-side generation now includes the official-style sliding no-repeat ngram
  processor, exposed through `--no-repeat-ngram-size` and `--ngram-window`.

Current no-cache prefill command:

```shell
python -m openvino_adapt.run_prefill_no_cache \
  --model-dir openvino_models/unlimited_ocr_prefill277 \
  --image path/to/page.png
```

This produces logits for the image+text prefill only. Use the explicit-KV
generation command below for R-SWA cached decoding.

Current explicit-KV generation command:

```shell
python -m openvino_adapt.run_generate_openvino \
  --image path/to/page.png \
  --max-new-tokens 8
```

Current OCR CLI command:

```shell
python -m openvino_adapt.run_ocr_openvino \
  --image path/to/page.png \
  --output-dir outputs_openvino \
  --max-new-tokens 128 \
  --no-repeat-ngram-size 35 \
  --ngram-window 128
```

Repeatable export/inspection commands:

```shell
python -m openvino_adapt.prompt_profile \
  --prompt "<image>document parsing." \
  --page-count 2

python -m openvino_adapt.export_all \
  --model models/Unlimited-OCR \
  --prompt "<image>document parsing." \
  --moe-impl dense \
  --fp16

python -m openvino_adapt.inspect_artifacts
```

## MoE export note

The upstream eval path uses `tokens_per_expert.cpu().numpy()` inside
`DeepseekV2MoE.moe_infer()`. A naive trace freezes the export-time expert
routing and gives wrong logits on real inputs. The adapter therefore uses a
dense tensorized MoE in `DecoderPrefillWithKV` and `DecoderDecodeOneStep`: all
experts are evaluated and masked by top-k weights. This is slower but preserves
correctness and keeps the OpenVINO graph input-dependent.

An experimental vectorized sparse top-k MoE was added behind `--moe-impl sparse`.
It matches PyTorch numerically and exports, but the tested OpenVINO CPU plugin
attempted to allocate hundreds of GB for a dynamic Gather during prefill. Keep
`--moe-impl dense` as the default until the sparse graph is rewritten or tested
on a plugin that handles that gather pattern efficiently.

Benchmark entry point:

```shell
python -m openvino_adapt.benchmark_openvino \
  --image path/to/page.png \
  --max-new-tokens 4
```

Compression entry point:

```shell
python -m openvino_adapt.compress_artifacts \
  --input-model openvino_models/unlimited_ocr_kv_dense_past677/decoder_decode_one.xml \
  --output-model openvino_models/unlimited_ocr_kv_dense_past677_int8/decoder_decode_one.xml \
  --mode int8_asym
```

Decode-only diagnostic entry point:

```shell
python -m openvino_adapt.benchmark_decode_one \
  --model openvino_models/unlimited_ocr_kv_dense_past677_int8/decoder_decode_one.xml \
  --device GPU \
  --cache-dir openvino_cache_decode_diag \
  --runs 1
```

Readiness doctor:

```shell
python -m openvino_adapt.doctor

python -m openvino_adapt.doctor \
  --image path/to/page.png \
  --run-one-token
```

Observed CPU dense-MoE benchmark on the smoke image:

- `max_new_tokens=1`
- compile time: about `38-56s`
- generation time: about `28-40s`
- throughput: about `0.025-0.036 tok/s`
- two-page FP16 continuous benchmark:
  - prompt tokens: `550`
  - compile time: about `21s`
  - generation time: about `24-32s`
  - throughput: about `0.031-0.042 tok/s`
- compressed two-page decode artifacts:
  - FP16 decode bin: `5597.56 MB`
  - INT8 decode bin: `2806.11 MB`
  - INT4_ASYM group-64 decode bin: `1654.99 MB`
- split-device benchmark with two smoke pages and INT8 decode:
  - CPU all, 2 tokens: about `0.048 tok/s`
  - CPU prefill + GPU decode, 2 tokens: about `0.063 tok/s`
  - GPU prefill + GPU decode, 2 tokens: about `0.134 tok/s`
  - GPU vision + GPU prefill + GPU decode, 8 tokens: about `0.973 tok/s`

The current practical fast path is INT8 decode plus GPU vision/prefill/decode
with `embed_tokens` left on CPU. INT4_ASYM group-64 compiles on CPU and is much
smaller on disk, but the tested GPU benchmark did not finish within 10 minutes.
The next structural milestone is still a host-dispatched sparse MoE decode path,
which should avoid evaluating all routed experts per token.

Decode-only device matrix clarified the INT4 issue:

- INT4 group-64 on CPU: compile about `5.2s` with cache, infer about `3.26s`.
- INT4 group-64 on `AUTO:GPU,CPU`: execution device was `(CPU)`, compile about
  `17.8s`, infer about `3.88s`.
- INT4 group-64 on `GPU`: timed out after `180s` during compile.
- INT4 group-64 on `HETERO:GPU,CPU`: timed out after `180s` during compile.
- INT8 on `GPU`: compile about `78.6s`, infer about `0.90s`.
- INT8 on `AUTO:GPU,CPU`: execution device `GPU.0`, compile about `7.65s`,
  infer about `0.82s`, but one subprocess produced a crash exit after writing
  valid JSON.
- INT8 on `HETERO:GPU,CPU`: execution device `GPU.0`, compile about `224.6s`,
  infer about `2.05s`.

Conclusion: OpenVINO GPU shared memory is not the blocker for INT4. The blocker
is the GPU plugin compile/lowering path for this INT4 MoE decode graph. Use INT8
for GPU acceleration; keep INT4 as a CPU-capable small artifact or future
compiler experiment.

## Sparse MoE Decode Progress

The first host-dispatched sparse MoE cut is working at layer granularity:

- `host_sparse_moe_forward()` dispatches only the selected routed experts in
  Python and matches the official MoE and dense-safe MoE exactly in FP32.
- Layer 0 is dense MLP; layers 1-11 are routed MoE.
- `export_sparse_moe_layer` exports gate, shared experts, and selected expert
  MLP subgraphs for one MoE layer.
- `run_sparse_moe_layer_openvino` ran layer 1 with experts
  `1, 3, 6, 9, 30, 36, 59`; the selected top-6 route matched PyTorch with max
  diff about `7.8e-7` on CPU.
- `DecodeMoEAttentionGate` splits a decode layer into:
  - attention residual,
  - MoE input after post-attention norm,
  - top-k expert ids and weights,
  - shared expert output,
  - new K/V cache tensors.
- `run_sparse_decode_layer_openvino` verified layer 1 end-to-end:
  OpenVINO attention/gate/shared + OpenVINO selected expert MLPs + residual add
  matched dense layer math with hidden max diff about `2.4e-7` and key/value max
  diff about `1.6e-6` when `attention_gate.xml` is FP32.

Important constraint: FP16-compressing the attention/gate graph can make K cache
drift too large on random inputs, even when hidden output still looks close. Keep
cache-producing sparse decode layer graphs FP32 for now; FP16 is acceptable for
expert MLP subgraphs in the current small test.

The full 12-layer sparse decode-one runtime is now implemented:

- `export_sparse_decode_all` emitted the complete `past_len=677` sparse decode
  artifact: 728 XML graphs under `openvino_models/sparse_decode_past677`.
- `run_sparse_decode_openvino` chains all layers, maintains the per-layer K/V
  outputs, lazily compiles only the routed expert MLP subgraphs, and applies
  final norm/head.
- CPU correctness against the PyTorch wrapper is good:
  - logits max absolute difference about `1.7e-5`,
  - K max absolute difference about `2.7e-6`,
  - V max absolute difference about `1.1e-6`.
- CPU selected expert inference across all MoE layers was about `0.46s` for one
  token after compilation; the CPU layer graphs were about `0.26s` total.
- CPU layer graphs plus GPU expert MLPs reduced expert inference to about
  `0.15s`, but the end-to-end logits/KV drift increased. This is useful as a
  fast approximate path, not yet the correctness baseline.
- All-GPU sparse decode ran, but cache drift was substantially larger. Keep
  attention/cache-producing sparse layer graphs FP32 on CPU until cache accuracy
  is solved.

Low-bit expert planning:

- `profile_sparse_experts` profiles expert route frequency and gate weight.
- `make_expert_precision_plan` turns that profile into a per-expert precision
  policy. The 4-sample smoke profile produced 66 hot `fp16` experts, 158
  `int8_asym` warm experts, and 480 `experimental_int2` unused-in-profile
  experts.
- The stock-OpenVINO aggressive variant uses the same hot/warm split but marks
  those 480 unused-in-profile experts as `fp4`; a dry-run confirmed coverage of
  all 704 routed experts.
- `compress_sparse_experts` can apply the plan to split sparse artifacts. It
  copies unsupported modes by default, so `experimental_int2` remains a marked
  future-kernel candidate while the artifact stays runnable.
- Current NNCF/OpenVINO modes include INT8, INT4, NF4, CB4, MXFP4, MXFP8, FP8,
  FP4, and NVFP4, but not INT2.
- The full stock FP4 mixed artifact was generated and smoke-tested:
  - base sparse artifact: `728` XML files, about `5949.05 MB`;
  - mixed FP4 artifact: `728` XML files, about `3362.05 MB`;
  - policy applied: 66 copied hot experts, 158 `int8_asym` warm experts, and
    480 `fp4` unused-in-profile experts.
- Mixed FP4 one-token CPU correctness against the PyTorch wrapper:
  - logits max absolute difference about `1.28e-2`;
  - K max absolute difference about `1.70e-3`;
  - V max absolute difference about `5.59e-4`.
- `run_ocr_openvino` and `benchmark_openvino` now support `--decoder sparse`
  with `--sparse-artifact-dir`. Sparse mode reuses the normal OpenVINO
  embed/vision/prefill graphs and routes decode tokens through the split sparse
  runtime.
- Two-page CPU generation benchmark with 4 generated tokens:
  - FP16 expert artifact: about `0.0491 tok/s`;
  - mixed FP4 artifact: about `0.0486 tok/s`.
  These include first-use lazy expert compilation, so they are smoke throughput
  numbers, not warmed steady-state numbers.
- Smoke tests completed for single-image dense OCR, two-page PDF continuous
  sparse OCR, and CPU sparse decode correctness.
- A single expert MLP compressed from `6.56 MB` to about `2.28 MB` with
  `int4_asym`, `2.24 MB` with `fp4`, and `2.23 MB` with `nf4`.

## Practical caution

This is now a functional research-grade OpenVINO adaptation: fixed-size
single-page image input, explicit-KV decoder prefill/decode, host-owned R-SWA
cache window, and OCR-style image/PDF runners all work. It is not yet a
comfortable production OCR engine because the correctness-preserving dense MoE
decoder is very slow and the sparse MoE graph needs a different OpenVINO-friendly
dispatch strategy.
