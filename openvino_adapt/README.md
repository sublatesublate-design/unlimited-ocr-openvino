# Unlimited-OCR OpenVINO 适配说明

这个目录是 Unlimited-OCR 的 OpenVINO 本地适配层。它不是简单地把模型整体导出成一个 `generate()` 图，而是把视觉编码、文本 embedding、decoder prefill、one-token decode、R-SWA cache 和 sparse MoE expert dispatch 拆开处理。

## 当前状态

已经完成：

- `embed_tokens` OpenVINO 图；
- `vision_tokens` OpenVINO 图；
- `decoder_prefill_kv` 显式 KV prefill 图；
- `decoder_decode_one` 显式 KV dense decode 图；
- host 侧 R-SWA / ring cache；
- 单图、图片目录、PDF OCR runner；
- two-page continuous OCR smoke；
- INT8 / INT4 dense decode artifact；
- 完整 12 层 host-dispatched sparse MoE decode；
- FP4 / INT8 mixed expert artifact；
- `run_ocr_openvino --decoder sparse`；
- CPU/GPU/AUTO/HETERO benchmark 和诊断脚本。

建议把当前版本视为 **v0.1 research adapter**。它已经可以上传 GitHub、复现实验、继续优化，但还不是完整产品化 OCR 系统。

## 安装

Windows 本地测试环境使用 Python 3.13。核心依赖：

```shell
python -m pip install torch transformers openvino nncf pymupdf pillow numpy safetensors
```

检查环境：

```shell
python -m openvino_adapt.env_check
```

下载模型到本地：

```shell
python -m openvino_adapt.download_model --local-dir models/Unlimited-OCR
```

模型权重不进 Git，需要在每台机器本地下载。

## 基础导出

单页默认 prompt：

```shell
python -m openvino_adapt.export_all ^
  --model models/Unlimited-OCR ^
  --prompt "<image>document parsing." ^
  --moe-impl dense ^
  --fp16
```

连续多页需要先确认 prompt token 数：

```shell
python -m openvino_adapt.prompt_profile ^
  --prompt "<image>document parsing." ^
  --page-count 2
```

本机已验证的两页配置：

- `prompt_tokens=550`
- `ring_window=128`
- `past_len=677`

对应关系是：

```text
past_len = prompt_tokens + ring_window - 1
```

## dense OCR 运行

单图：

```shell
python -m openvino_adapt.run_ocr_openvino ^
  --image path/to/page.png ^
  --output-dir outputs_openvino_single ^
  --max-new-tokens 128 ^
  --cache-dir openvino_cache
```

PDF 按页独立 OCR：

```shell
python -m openvino_adapt.run_ocr_openvino ^
  --pdf path/to/document.pdf ^
  --output-dir outputs_openvino_pdf ^
  --max-new-tokens 128 ^
  --cache-dir openvino_cache
```

两页 continuous dense OCR：

```shell
python -m openvino_adapt.run_ocr_openvino ^
  --pdf path/to/two_page.pdf ^
  --continuous ^
  --prompt "<image>document parsing." ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --decode-model openvino_models/unlimited_ocr_kv_dense_past677/decoder_decode_one.xml ^
  --output-dir outputs_openvino_two_page_dense ^
  --max-new-tokens 128
```

输出包括：

- `page_*.md`
- `combined.md`
- `manifest.json`
- continuous 模式下还有 `continuous.md`

## sparse MoE decode 导出

导出完整 12 层 sparse decode artifact：

```shell
python -m openvino_adapt.export_sparse_decode_all ^
  --output-dir openvino_models/sparse_decode_past677 ^
  --past-len 677 ^
  --expert-fp16 ^
  --skip-existing
```

结构：

- layer 0：dense decode layer；
- layers 1-11：`attention_gate.xml`、`add_moe_residual.xml`、64 个 expert MLP；
- `final_norm_head.xml`。

完整 artifact 共 728 个 XML。

## expert profile 和 mixed FP4

生成专家路由统计：

```shell
python -m openvino_adapt.profile_sparse_experts ^
  --samples 4 ^
  --output-json outputs_openvino_ngram_smoke/expert_route_profile_4samples.json
```

生成 stock OpenVINO mixed precision 计划：

```shell
python -m openvino_adapt.make_expert_precision_plan ^
  --profile-json outputs_openvino_ngram_smoke/expert_route_profile_4samples.json ^
  --output-json outputs_openvino_ngram_smoke/expert_precision_plan_4samples_stock_fp4.json ^
  --unused-mode fp4
```

应用计划：

```shell
python -m openvino_adapt.compress_sparse_experts ^
  --artifact-dir openvino_models/sparse_decode_past677 ^
  --plan-json outputs_openvino_ngram_smoke/expert_precision_plan_4samples_stock_fp4.json ^
  --output-dir openvino_models/sparse_decode_past677_mixed_fp4
```

本机 4-sample profile 的计划：

- 66 个 hot experts：复制保留；
- 158 个 warm experts：`int8_asym`；
- 480 个 unused-in-profile experts：`fp4`。

## sparse OCR 运行

两页 PDF continuous sparse OCR：

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

注意：sparse artifact 是固定 shape 的。`past_len=677` 必须匹配：

```text
prompt_tokens + ring_window - 1 == 677
```

如果 shape 不匹配，runner 会直接报错。

## correctness 检查

检查 mixed FP4 sparse decode 和 PyTorch wrapper 的 one-token 对齐：

```shell
python -m openvino_adapt.run_sparse_decode_openvino ^
  --artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --device CPU ^
  --past-len 677 ^
  --cache-dir openvino_cache_smoke/sparse_mixed_fp4 ^
  --output-json outputs_openvino_ngram_smoke/sparse_decode_mixed_fp4_cpu_cache.json
```

本机结果：

- logits max diff 约 `1.28e-2`；
- K max diff 约 `1.70e-3`；
- V max diff 约 `5.59e-4`。

FP16 expert artifact 的 CPU baseline：

- logits max diff 约 `1.7e-5`；
- K max diff 约 `2.7e-6`；
- V max diff 约 `1.1e-6`。

## benchmark

两页 sparse generation benchmark：

```shell
python -m openvino_adapt.benchmark_openvino ^
  --image outputs_openvino_2page_input/page_0001.png outputs_openvino_2page_input/page_0002.png ^
  --prompt "<image>document parsing." ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --decoder sparse ^
  --sparse-artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --device CPU ^
  --max-new-tokens 4 ^
  --output-json outputs_openvino_ngram_smoke/benchmark_sparse_mixed_fp4_cpu_2page_4tok.json
```

本机 cold smoke throughput：

- FP16 expert artifact：约 `0.0491 tok/s`；
- mixed FP4 artifact：约 `0.0486 tok/s`。

这些数字包含首次 lazy compile 66 个专家，不代表 warm steady-state。

## 设备策略

dense decode 已测结果：

- INT8 GPU decode 可以跑；
- INT4 CPU 可以编译和推理；
- INT4 GPU / HETERO 在本机编译阶段超时；
- `AUTO:GPU,CPU` 对 INT4 最终落到 CPU；
- 当前实际可用的 GPU 快速路径是 INT8，不是 INT4。

sparse decode 已测结果：

- CPU 是 correctness baseline；
- CPU layer + GPU experts 可以加速 expert 部分，但误差变大；
- all-GPU sparse decode cache drift 过大，暂时不作为基准路径。

## OpenVINO cache

所有 runner 支持：

```shell
--cache-dir openvino_cache
```

cache 和本机硬件、驱动、OpenVINO 版本相关，不应该提交到 Git。

## 已知限制

- 这是研究版适配，不是完整 OCR 产品。
- sparse decode 当前需要固定 shape。
- 不同页数、prompt、ring window 需要匹配导出。
- 本地 OpenVINO/NNCF 没有 INT2 weight compression。
- `experimental_int2` 只是未来自定义 kernel 的标记。
- PDF runner 的 `manifest.json` 里页图路径可能是临时渲染路径。
- 稳定输出是 `continuous.md`、`page_*.md`、`combined.md` 和 `manifest.json`。

## 文件说明

- `export_openvino.py`：单组件导出。
- `export_all.py`：基础 artifact 批量导出。
- `run_ocr_openvino.py`：图像/PDF OCR runner。
- `run_generate_openvino.py`：OpenVINO greedy generation loop。
- `run_sparse_decode_openvino.py`：完整 12 层 sparse decode runtime。
- `export_sparse_decode_all.py`：完整 sparse decode artifact 导出。
- `profile_sparse_experts.py`：专家路由统计。
- `make_expert_precision_plan.py`：生成 per-expert precision plan。
- `compress_sparse_experts.py`：按计划压缩 sparse experts。
- `benchmark_openvino.py`：端到端生成 benchmark。
- `benchmark_decode_one.py`：decode graph 诊断 benchmark。
- `runtime.py`：OpenVINO device/cache 编译辅助。
