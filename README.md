# Unlimited-OCR OpenVINO 适配版

这是基于百度 `Unlimited-OCR` 的本地 OpenVINO 研究适配版，重点是让模型在 Intel CPU/GPU 环境中跑通 OCR 推理、连续多页输入、显式 KV 缓存、稀疏 MoE 解码和混合精度专家压缩。

上游项目：

- 官方 GitHub：`https://github.com/baidu/Unlimited-OCR`
- 官方模型：`https://huggingface.co/baidu/Unlimited-OCR`
- 论文：`https://arxiv.org/abs/2606.23050`

本适配版的预生成混合 FP4 OpenVINO 产物：

- Hugging Face：`https://huggingface.co/sublatesublate-design/unlimited-ocr-openvino`

本仓库不直接托管模型权重，也不提交 OpenVINO IR、缓存和 OCR 输出。这些文件体积很大，而且通常需要按本机硬件、OpenVINO 版本、页数和提示词重新生成。

## 已实现内容

- OpenVINO `embed_tokens`、`vision_tokens`、`decoder_prefill_kv`、`decoder_decode_one` 导出。
- 显式 KV 预填充 / 解码推理路径。
- 主机侧 R-SWA / 环形缓存管理。
- 单图、图片目录、PDF OCR 运行器。
- 多页连续 OCR 路径。
- INT8、INT4、FP4、NF4 等 NNCF 权重量化实验。
- 完整 12 层主机调度稀疏 MoE 解码运行时。
- 稀疏专家重要性统计。
- 混合 FP4/INT8 专家产物生成。
- fused-hot-gather 单层融合稀疏解码。
- 多层 decoder block 融合导出与运行时接入。
- `run_ocr_openvino --decoder sparse` 正式 CLI 接入。
- CPU/GPU/AUTO/HETERO 诊断基准测试。
- 中文快速开始和冒烟测试说明。

## 当前定位

这是 **v0.1 研究适配版**，不是完整产品化 OCR 套件。

可以用于：

- 复现 Unlimited-OCR 的 OpenVINO 适配路径；
- 研究 R-SWA 缓存如何在 OpenVINO 外部调度；
- 研究稀疏 MoE 解码如何由主机侧调度；
- 研究 OpenVINO/NNCF 下 INT8、INT4、FP4 混合精度；
- 作为后续本地 OCR 产品化系统的底座。

暂时不承诺：

- 高吞吐生产 OCR；
- 任意页数/任意提示词的通用产物；
- 全 GPU 稀疏解码的数值稳定性；
- INT2 原生 OpenVINO 压缩；
- 原始、清理、校对三层文档交付。

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

下载预生成混合 FP4 稀疏解码产物：

```shell
hf download sublatesublate-design/unlimited-ocr-openvino ^
  --repo-type model ^
  --local-dir openvino_models/sparse_decode_past677_mixed_fp4
```

导出基础 OpenVINO 图：

```shell
python -m openvino_adapt.export_all ^
  --model models/Unlimited-OCR ^
  --prompt "<image>document parsing." ^
  --moe-impl dense ^
  --fp16
```

两页连续稀疏 OCR 示例：

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
  --sparse-artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --sparse-device GPU ^
  --sparse-expert-device GPU ^
  --sparse-hot-pack-dir openvino_models/hot_expert_packs_current_2page_2tok_v4_fp16 ^
  --sparse-hot-pack-device GPU ^
  --sparse-precompile-static ^
  --max-new-tokens 128
```

上面这条是本机冒烟测试路径。`hot_expert_packs_current_2page_2tok_v4_fp16` 是针对当前两页测试输入生成的样本热包；换 PDF、页数、提示词或长输出时，需要重新统计路由并生成带安全余量的热包。

调试时建议先不要传 `--cache-dir`，或者只使用一个明确的小缓存目录。OpenVINO 会为不同图和设备生成大量编译缓存，长时间基准测试可能让磁盘快速膨胀。

## 已验证结果

本机 Windows + OpenVINO 2025.4.1 环境下：

- 完整稀疏解码产物：728 个 XML，约 `5949.05 MB`。
- 混合 FP4 产物：728 个 XML，约 `3362.05 MB`。
- 混合精度策略：66 个热专家保留，158 个 `int8_asym`，480 个 `fp4`。
- 混合 FP4 CPU 单 token 正确性：
  - logits max diff 约 `1.28e-2`；
  - K max diff 约 `1.70e-3`；
  - V max diff 约 `5.59e-4`。
- 两页 CPU 稀疏生成，4 tokens：
  - FP16 专家产物约 `0.0491 tok/s`；
  - 混合 FP4 产物约 `0.0486 tok/s`。

这些速度包含首次延迟编译 66 个专家，是冷启动冒烟吞吐，不是热启动稳态吞吐。

进一步的热专家包诊断：

- `top16` 通用热包覆盖不足：首个稀疏解码步仍有 `58` 次回退专家，约 `7.01 s/step`。
- 当前两页样本专用 FP16 热包 v2：`0` 次回退专家，约 `2.36 s/step`，对应稀疏解码约 `0.42 tok/s`。
- 同一 v2 热包放到 GPU：热包自身从约 `0.59 s` 降到约 `0.44 s`，但整步约 `2.68 s/step`，因为注意力、门控、KV、最终头仍主要在 CPU 路径，跨设备收益没有闭环。
- 当前两页样本专用 FP16 热包 v4 + 全 GPU：`0` 次回退专家，稀疏解码约 `0.36 s/step`，对应约 `2.78 tok/s`；两页冒烟测试端到端 `decode_seconds` 约 `4.36-4.75 s`。
- 受控 OpenVINO 缓存：`openvino_cache_controlled/full_gpu_v4` 当前约 `55.90 GB`。同一配置热启动编译可从约 `72.5 s` 降到约 `20.5-36.6 s`，取决于本轮是否新增图和 Windows 文件锁状态。
- 12-token 强制长输出统计：v4 热包有 `610` 次回退，解码循环约 `38.28 s`；top48 v1 热包降到 `3` 次回退，解码循环约 `2.13 s`，稀疏解码约 `5.19 tok/s`；top48 v2 热包进一步降到 `0` 次回退，解码循环约 `2.09 s`，稀疏解码约 `5.29 tok/s`。
- 32-token 强制长输出统计：top48 v2 又出现 `381` 次回退，解码循环约 `28.78 s`；top61 gather-pack 降到 `0` 次回退，解码循环约 `4.32 s`；fused-hot-gather 继续降到 `3.62 s`，稀疏解码约 `0.116 s/step`。
- 常驻 JSONL 服务：同一进程内复用已编译 OpenVINO 图。top48 v2 两次 12-token 任务中第二次总耗时约 `5.45 s`，稀疏解码约 `7.04 tok/s`；fused-hot-gather 两次 32-token 任务中第二次总耗时约 `6.62 s`，解码循环约 `2.86 s`，整体约 `4.85 tok/s`，`0` 次回退。它适合把编译成本摊到连续 OCR 任务里。
- 多层 decoder block 融合已经接入，但不是当前最快稳定路径。block2 第二轮约 `3.11 tok/s`，block3 常驻第二任务约 `4.54 tok/s`，block4 常驻第二任务约 `4.59 tok/s`；block6 首次解码循环较快，但第二任务触发 Intel GPU `CL_OUT_OF_RESOURCES`，暂不推荐。
- final head 的 `TopK(16)` 和 `ArgMax` 快路径已经接入为实验开关，但本机 OpenVINO GPU 实测更慢：TopK/ArgMax 每步约 `18-20 ms`，完整 logits final head 通常约 `6-7 ms`。
- 短 KV 窗口实验没有超过当前推荐配置：`ring_window=64 / past_len=613` 常驻第二任务约 `4.57 tok/s`，`ring_window=32 / past_len=581` 常驻第二任务约 `3.94 tok/s`。

当前结论：最快稳定路径是基础组件和融合稀疏解码全部放 GPU，并配合按真实路由生成的 fused-hot-gather 单层图。`fused_hot_gather_past677_top61_fp16` 是目前最稳的两页 32-token 样本稀疏解码产物，约 `4.89 GB`；换 PDF、页数、提示词或输出长度时，需要重新统计路由并生成新的融合产物。

受控缓存管理：

```shell
python -m openvino_adapt.manage_openvino_cache ^
  --cache-dir openvino_cache_controlled/full_gpu_v4 ^
  --max-gb 12
```

如果 Windows 暂时锁住部分 OpenVINO `.blob`，工具会跳过并报告；稍后重试即可。

常驻进程示例：

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

启动后向标准输入写入 JSONL 任务，例如：

```json
{"id":"job-1","images":["outputs_openvino_2page_input/page_0001.png","outputs_openvino_2page_input/page_0002.png"],"max_new_tokens":32,"eos_token_id":-1}
```

## 仓库内容

- `openvino_adapt/`：OpenVINO 导出、运行、量化、基准测试、稀疏 MoE 运行时。
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
