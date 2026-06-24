# Unlimited-OCR OpenVINO 适配版

这是基于百度 `Unlimited-OCR` 的本地 OpenVINO 研究适配版，重点是让模型在 Intel CPU/GPU 环境中跑通 OCR 推理、连续多页输入、显式 KV cache、sparse MoE decode 和 mixed precision expert 压缩。

上游项目：

- 官方 GitHub：`https://github.com/baidu/Unlimited-OCR`
- 官方模型：`https://huggingface.co/baidu/Unlimited-OCR`
- 论文：`https://arxiv.org/abs/2606.23050`

本仓库不直接托管模型权重，也不提交 OpenVINO IR、cache 和 OCR 输出。这些文件体积很大，而且通常需要按本机硬件、OpenVINO 版本、页数和 prompt 重新生成。

## 已实现内容

- OpenVINO `embed_tokens`、`vision_tokens`、`decoder_prefill_kv`、`decoder_decode_one` 导出。
- 显式 KV prefill / decode 推理路径。
- host 侧 R-SWA / ring cache 管理。
- 单图、图片目录、PDF OCR runner。
- 多页 continuous OCR 路径。
- INT8、INT4、FP4、NF4 等 NNCF 权重量化实验。
- 完整 12 层 host-dispatched sparse MoE decode runtime。
- sparse expert importance profile。
- mixed FP4/INT8 expert artifact 生成。
- `run_ocr_openvino --decoder sparse` 正式 CLI 接入。
- CPU/GPU/AUTO/HETERO 诊断 benchmark。
- 中文快速开始和 smoke test 说明。

## 当前定位

这是 **v0.1 research adapter**，不是完整产品化 OCR 套件。

可以用于：

- 复现 Unlimited-OCR 的 OpenVINO 适配路径；
- 研究 R-SWA cache 如何在 OpenVINO 外部调度；
- 研究 sparse MoE decode 如何 host-dispatch；
- 研究 OpenVINO/NNCF 下 INT8、INT4、FP4 mixed precision；
- 作为后续本地 OCR 产品化系统的底座。

暂时不承诺：

- 高吞吐生产 OCR；
- 任意页数/任意 prompt 的通用 artifact；
- 全 GPU sparse decode 的数值稳定性；
- INT2 原生 OpenVINO 压缩；
- raw / cleaned / reviewed 三层文档交付。

## 快速开始

详细命令在 [openvino_adapt/README.md](openvino_adapt/README.md)。

安装核心依赖：

```shell
python -m pip install torch transformers openvino nncf pymupdf pillow numpy safetensors
```

下载模型：

```shell
python -m openvino_adapt.download_model --local-dir models/Unlimited-OCR
```

导出基础 OpenVINO 图：

```shell
python -m openvino_adapt.export_all ^
  --model models/Unlimited-OCR ^
  --prompt "<image>document parsing." ^
  --moe-impl dense ^
  --fp16
```

两页 continuous sparse OCR 示例：

```shell
python -m openvino_adapt.run_ocr_openvino ^
  --pdf path/to/two_page.pdf ^
  --pdf-dpi 72 ^
  --output-dir outputs_openvino_sparse_pdf ^
  --continuous ^
  --prompt "<image>document parsing." ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --decoder sparse ^
  --sparse-artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --max-new-tokens 128 ^
  --cache-dir openvino_cache
```

## 已验证结果

本机 Windows + OpenVINO 2025.4.1 环境下：

- 完整 sparse decode artifact：728 个 XML，约 `5949.05 MB`。
- mixed FP4 artifact：728 个 XML，约 `3362.05 MB`。
- mixed policy：66 个 hot experts 保留，158 个 `int8_asym`，480 个 `fp4`。
- mixed FP4 CPU one-token correctness：
  - logits max diff 约 `1.28e-2`；
  - K max diff 约 `1.70e-3`；
  - V max diff 约 `5.59e-4`。
- 两页 CPU sparse generation，4 tokens：
  - FP16 expert artifact 约 `0.0491 tok/s`；
  - mixed FP4 artifact 约 `0.0486 tok/s`。

这些速度包含首次 lazy compile 66 个专家，是 cold smoke throughput，不是 warm steady-state throughput。

## 仓库内容

- `openvino_adapt/`：OpenVINO 导出、运行、量化、benchmark、sparse MoE runtime。
- `research/UNLIMITED_OCR_OPENVINO_NOTES.md`：研究记录和实测结果。
- `research/inspect_unlimitedocr.py`：模型结构检查脚本。
- `research/pytorch_single_page_baseline.py`：PyTorch baseline 辅助脚本。
- `infer.py`：上游 SGLang/PyTorch 推理入口保留文件。

## 不提交的内容

`.gitignore` 已排除：

- `models/`
- `openvino_models/`
- `openvino_cache*/`
- `outputs*/`
- `*.safetensors`
- `*.onnx`
- `*.gguf`
- `*.bin`

这些需要按 README 在本地重新下载、导出或生成。

## 许可证

本仓库保留上游 `LICENSE`。模型权重、论文、数据和上游代码的使用条件请同时遵守百度 Unlimited-OCR 官方仓库和模型页面的许可说明。
