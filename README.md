# Unlimited-OCR OpenVINO: A Research Adapter for Sparse MoE OCR Decoding

这是一个面向 Intel CPU/GPU 的 **Unlimited-OCR OpenVINO 研究适配版**。它不是简单地把模型导出成一个 OpenVINO `generate()` 图，而是把 Unlimited-OCR 的视觉编码、预填充、R-SWA 缓存、12 层稀疏 MoE 解码、专家路由和长输出基准测试拆开实现，并在本地 Windows + Intel GPU 环境中验证可运行路径。

上游项目：

- 官方 GitHub：`https://github.com/baidu/Unlimited-OCR`
- 官方模型：`https://huggingface.co/baidu/Unlimited-OCR`
- 论文：`https://arxiv.org/abs/2606.23050`

本项目产物：

- GitHub 代码：`https://github.com/sublatesublate-design/unlimited-ocr-openvino`
- Hugging Face OpenVINO 产物：`https://huggingface.co/sublatesublate-design/unlimited-ocr-openvino`

## Abstract

Unlimited-OCR 使用 VLM + R-SWA 长上下文机制来做长文档 OCR。官方推理路径主要面向 PyTorch / vLLM / SGLang / CUDA 生态；直接迁移到 OpenVINO 会遇到三个核心问题：自定义模型结构、R-SWA KV 缓存调度、以及 sparse MoE decoder 的专家路由。

本仓库给出一个工程化适配方案：将模型拆成 OpenVINO 可编译的子图，在 Python 侧维护 R-SWA/ring cache，并实现完整 12 层 sparse MoE decode runtime。进一步地，本仓库实现了 mixed FP4/INT8 压缩、hot expert pack、top-k gather、fused-hot-gather、常驻 JSONL 服务、OpenVINO cache 管理和长输出路由 profile。最终在本机两页 32-token smoke benchmark 中，最快稳定路径达到约 `4.85 tok/s`。

这仍然不是生产级 OCR 套件，但它证明了 Unlimited-OCR 可以被拆解、量化、部分融合，并在非 CUDA 的 OpenVINO Runtime 中运行。

## Contributions

本仓库主要完成了以下工作：

- **OpenVINO 拆图导出**：`embed_tokens`、`vision_tokens`、`decoder_prefill_kv`、dense decode、sparse decode layer、final head。
- **显式 KV cache runtime**：在 OpenVINO 外部维护 R-SWA / ring cache，支持连续多页输入。
- **完整 12 层 sparse MoE decode**：支持主机侧专家调度、fallback expert、hot expert pack、fused-hot-gather。
- **混合精度实验**：支持 INT8、INT4、FP4、NF4 等 NNCF 压缩实验，并给出 mixed FP4/INT8 expert artifact。
- **GPU 路径优化**：实现 all-GPU sparse decode、top-k gather pack、layer fused-hot-gather、常驻服务复用编译图。
- **负结果记录**：验证并记录 block fusion、TopK/ArgMax final head、短 KV window 等路线在本机没有超过当前最佳路径。
- **工程化 CLI**：`run_ocr_openvino --decoder sparse`、`benchmark_openvino`、`serve_ocr_openvino`、cache 管理和 benchmark 汇总工具。

## Method

Unlimited-OCR 的困难不在普通图导出，而在 decoder 状态调度。朴素路线通常是：

```text
PDF / image -> vision encoder -> language model generate()
```

本仓库采用拆图式 runtime：

```text
images
  -> OpenVINO vision_tokens
  -> OpenVINO decoder_prefill_kv
  -> Python R-SWA / ring KV cache
  -> OpenVINO sparse decode layers
       -> attention + gate
       -> hot expert / fallback expert / fused-hot-gather
  -> OpenVINO final_norm_head
  -> Markdown text
```

核心设计点：

- `prefill` 阶段生成每层 K/V。
- `decode` 阶段每次只生成一个 token。
- `ref/prompt KV` 保留，输出 token KV 通过固定窗口滚动。
- sparse MoE 层可以在三种模式之间切换：
  - 主机侧路由 + 单专家图；
  - hot expert pack；
  - fused-hot-gather 单层融合图。
- 常驻服务模式复用 OpenVINO 已编译图，避免每个任务重复 compile。

## Current Best Artifact

当前推荐的最快稳定产物是：

```text
fused_hot_gather_past677_top61_fp16/
```

它已经上传到 Hugging Face：

```shell
hf download sublatesublate-design/unlimited-ocr-openvino ^
  --repo-type model ^
  --include "fused_hot_gather_past677_top61_fp16/*" ^
  --local-dir openvino_models
```

下载后 artifact 路径为：

```text
openvino_models/fused_hot_gather_past677_top61_fp16
```

注意：Hugging Face 仓库根目录仍保留早期 sparse/mixed artifact。推荐新实验优先使用 `fused_hot_gather_past677_top61_fp16/` 子目录。

## Quick Start

安装核心依赖：

```shell
python -m pip install torch transformers openvino nncf pymupdf pillow numpy safetensors huggingface_hub
```

下载上游模型：

```shell
python -m openvino_adapt.download_model --local-dir models/Unlimited-OCR
```

下载推荐 OpenVINO artifact：

```shell
hf download sublatesublate-design/unlimited-ocr-openvino ^
  --repo-type model ^
  --include "fused_hot_gather_past677_top61_fp16/*" ^
  --local-dir openvino_models
```

两页连续 OCR / generation 示例：

```shell
python -m openvino_adapt.run_ocr_openvino ^
  --pdf path/to/two_page.pdf ^
  --pdf-dpi 72 ^
  --output-dir outputs_openvino_sparse_pdf ^
  --continuous ^
  --device GPU ^
  --prompt "<image>document parsing." ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --decoder sparse ^
  --sparse-artifact-dir openvino_models/fused_hot_gather_past677_top61_fp16 ^
  --sparse-device GPU ^
  --sparse-expert-device GPU ^
  --sparse-precompile-static ^
  --max-new-tokens 128
```

常驻服务示例：

```shell
python -m openvino_adapt.serve_ocr_openvino ^
  --decoder sparse ^
  --device GPU ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --sparse-artifact-dir openvino_models/fused_hot_gather_past677_top61_fp16 ^
  --sparse-device GPU ^
  --sparse-expert-device GPU ^
  --sparse-precompile-static ^
  --cache-dir openvino_cache_controlled/full_gpu_v4 ^
  --max-new-tokens 32 ^
  --eos-token-id -1
```

服务启动后向标准输入写入 JSONL：

```json
{"id":"job-1","images":["outputs_openvino_2page_input/page_0001.png","outputs_openvino_2page_input/page_0002.png"],"max_new_tokens":32,"eos_token_id":-1}
```

更多导出、profile、benchmark 命令见 [openvino_adapt/README.md](openvino_adapt/README.md)。

## Experimental Results

测试环境：

- Windows
- OpenVINO `2025.4.1`
- Intel GPU
- 两页输入，`prompt_tokens=550`
- `ring_window=128`
- `past_len=677`

### Correctness

| Artifact | Device | Metric | Result |
| --- | --- | --- | --- |
| FP16 sparse expert | CPU | logits max diff | `~1.7e-5` |
| FP16 sparse expert | CPU | K max diff | `~2.7e-6` |
| FP16 sparse expert | CPU | V max diff | `~1.1e-6` |
| mixed FP4 | CPU | logits max diff | `~1.28e-2` |
| mixed FP4 | CPU | K max diff | `~1.70e-3` |
| mixed FP4 | CPU | V max diff | `~5.59e-4` |

### Sparse Decode Speed

| Configuration | Fallback experts | Decode loop / sparse step | Throughput |
| --- | ---: | ---: | ---: |
| CPU FP16 sparse, 4 tokens | lazy compile | cold smoke | `~0.049 tok/s` |
| CPU mixed FP4, 4 tokens | lazy compile | cold smoke | `~0.049 tok/s` |
| sample hot pack v4, all GPU, 2 tokens | `0` | `~0.36 s/step` | `~2.78 tok/s` |
| top48 long-profile, 12 tokens | `3` | `~2.13 s loop` | `~5.19 sparse tok/s` |
| top48 v2, 12 tokens | `0` | `~2.09 s loop` | `~5.29 sparse tok/s` |
| top48 v2, 32 tokens | `381` | `~28.78 s loop` | `~0.91 tok/s` |
| top61 gather-pack, 32 tokens | `0` | `~4.32 s loop` | `~3.10 tok/s` |
| fused-hot-gather, 32 tokens | `0` | `~3.62 s loop` | `~3.51 tok/s` |
| fused-hot-gather persistent server, second 32-token job | `0` | `~2.86 s loop` | `~4.85 tok/s` |

### Negative Results

这些路线已经实现并测试，但不作为当前推荐路径：

| Experiment | Result | Interpretation |
| --- | --- | --- |
| decoder block2 fusion | `~3.11 tok/s` | 慢于单层 fused-hot-gather |
| decoder block3 fusion | persistent second job `~4.54 tok/s` | 仍慢于当前推荐 |
| decoder block4 fusion | persistent second job `~4.59 tok/s` | 稳定但不够快 |
| decoder block6 fusion | second job `CL_OUT_OF_RESOURCES` | Intel GPU 资源压力过大 |
| `TopK(16)` final head | `~18-20 ms/step` | 慢于完整 logits final head |
| `ArgMax` final head | `~18-20 ms/step` | 归约开销抵消回传收益 |
| `ring_window=64 / past_len=613` | persistent second job `~4.57 tok/s` | 没超过 `past677` |
| `ring_window=32 / past_len=581` | persistent second job `~3.94 tok/s` | 明显变慢 |

## What This Is Not

这不是完整产品化 OCR 套件。当前版本不承诺：

- 高吞吐批量扫书；
- 任意页数、任意 prompt 的通用 artifact；
- 全 GPU sparse decode 在所有输入上的数值稳定性；
- OpenVINO 原生 INT2；
- 原始、清理、校对三层文档交付；
- 与 CUDA/SGLang 官方路径同等速度。

当前更准确的定位是：

```text
Unlimited-OCR -> OpenVINO research adapter
```

它证明了一条可运行路线，也暴露了 Intel GPU + OpenVINO 上 sparse MoE VLM decoder 的实际瓶颈。

## Repository Layout

- `openvino_adapt/`：OpenVINO 导出、运行、量化、benchmark、sparse MoE runtime。
- `openvino_adapt/serve_ocr_openvino.py`：常驻 JSONL 服务，复用编译图。
- `openvino_adapt/export_sparse_decode_all.py`：完整 sparse decode artifact 导出。
- `openvino_adapt/export_fused_decode_blocks.py`：多层 decoder block 融合实验。
- `openvino_adapt/manage_openvino_cache.py`：OpenVINO cache 统计和裁剪。
- `research/UNLIMITED_OCR_OPENVINO_NOTES.md`：研究记录和实测结果。
- `infer.py`：上游 SGLang/PyTorch 推理入口保留文件。

## Files Not Tracked by Git

`.gitignore` 已排除：

- `models/`
- `openvino_models/`
- `openvino_cache*/`
- `outputs*/`
- `*.safetensors`
- `*.onnx`
- `*.gguf`
- `*.bin`

这些内容体积大、与本机硬件和 OpenVINO 版本强相关，应该通过 Hugging Face 下载或在本地重新导出。

## Citation

如果引用本项目，请同时引用上游 Unlimited-OCR 项目。本仓库是 OpenVINO 适配和工程实验，不替代上游模型、论文和许可证声明。

```bibtex
@misc{unlimited_ocr_openvino_adapter,
  title        = {Unlimited-OCR OpenVINO: A Research Adapter for Sparse MoE OCR Decoding},
  author       = {sublatesublate-design},
  year         = {2026},
  howpublished = {GitHub and Hugging Face},
  url          = {https://github.com/sublatesublate-design/unlimited-ocr-openvino}
}
```

## License

本仓库保留上游 `LICENSE`。模型权重、论文、数据和上游代码的使用条件请同时遵守百度 Unlimited-OCR 官方仓库和模型页面的许可说明。
