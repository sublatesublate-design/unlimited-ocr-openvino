# Unlimited-OCR OpenVINO 适配说明

这个目录是 Unlimited-OCR 的 OpenVINO 本地适配层。它不是简单地把模型整体导出成一个 `generate()` 图，而是把视觉编码、文本嵌入、decoder 预填充、单 token 解码、R-SWA 缓存和稀疏 MoE 专家调度拆开处理。

## 当前状态

已经完成：

- `embed_tokens` OpenVINO 图；
- `vision_tokens` OpenVINO 图；
- `decoder_prefill_kv` 显式 KV 预填充图；
- `decoder_decode_one` 显式 KV 稠密解码图；
- 主机侧 R-SWA / 环形缓存；
- 单图、图片目录、PDF OCR 运行器；
- 两页连续 OCR 冒烟测试；
- INT8 / INT4 稠密解码产物；
- 完整 12 层主机调度稀疏 MoE 解码；
- FP4 / INT8 混合专家产物；
- fused-hot-gather 单层融合稀疏解码；
- 多层 decoder block 融合导出与运行时接入；
- `run_ocr_openvino --decoder sparse`；
- CPU/GPU/AUTO/HETERO 基准测试和诊断脚本。

建议把当前版本视为 **v0.1 研究适配版**。它已经可以上传 GitHub、复现实验、继续优化，但还不是完整产品化 OCR 系统。

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

单页默认提示词：

```shell
python -m openvino_adapt.export_all ^
  --model models/Unlimited-OCR ^
  --prompt "<image>document parsing." ^
  --moe-impl dense ^
  --fp16
```

连续多页需要先确认提示词 token 数：

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

## 稠密 OCR 运行

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

两页连续稠密 OCR：

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
- 连续模式下还有 `continuous.md`

## 稀疏 MoE 解码导出

导出完整 12 层稀疏解码产物：

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

完整产物共 728 个 XML。

## 专家统计和混合 FP4

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

本机 4-sample 路由统计的计划：

- 66 个 hot experts：复制保留；
- 158 个 warm experts：`int8_asym`；
- 480 个统计中未使用专家：`fp4`。

## sparse OCR 运行

两页 PDF 连续稀疏 OCR：

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

`hot_expert_packs_current_2page_2tok_v4_fp16` 是针对当前两页冒烟测试输入生成的样本热包。正式 OCR 时应先用目标文档或相近样本生成路由统计，再导出热包；否则会退回单专家图，速度会明显下降。

注意：稀疏产物是固定 shape 的。`past_len=677` 必须匹配：

```text
prompt_tokens + ring_window - 1 == 677
```

如果 shape 不匹配，运行器会直接报错。

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

FP16 专家产物的 CPU 基线：

- logits max diff 约 `1.7e-5`；
- K max diff 约 `2.7e-6`；
- V max diff 约 `1.1e-6`。

## 基准测试

两页稀疏生成基准测试：

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

本机冷启动冒烟吞吐：

- FP16 专家产物：约 `0.0491 tok/s`；
- 混合 FP4 产物：约 `0.0486 tok/s`。

这些数字包含首次延迟编译 66 个专家，不代表热启动稳态。

全 GPU + 热专家包冒烟测试：

```shell
python -m openvino_adapt.benchmark_openvino ^
  --decoder sparse ^
  --device GPU ^
  --image outputs_openvino_2page_input/page_0001.png outputs_openvino_2page_input/page_0002.png ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --sparse-artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --sparse-device GPU ^
  --sparse-expert-device GPU ^
  --sparse-hot-pack-dir openvino_models/hot_expert_packs_current_2page_2tok_v4_fp16 ^
  --sparse-hot-pack-device GPU ^
  --sparse-precompile-static ^
  --max-new-tokens 2 ^
  --output-json outputs_openvino_sparse_logs/benchmark_mixed_fp4_samplepack_v4_full_gpu_precompile_2tok.json
```

本机结果：

- `0` 次回退专家；
- 稀疏解码约 `0.36 s/step`，约 `2.78 tok/s`；
- 两页冒烟测试端到端 `decode_seconds` 约 `4.36-4.75 s`；
- 冷启动编译仍然较慢，约 `50 s`，需要受控缓存或常驻进程摊销。

受控缓存复用：

```shell
python -m openvino_adapt.benchmark_openvino ^
  --decoder sparse ^
  --device GPU ^
  --image outputs_openvino_2page_input/page_0001.png outputs_openvino_2page_input/page_0002.png ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --sparse-artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --sparse-device GPU ^
  --sparse-expert-device GPU ^
  --sparse-hot-pack-dir openvino_models/hot_expert_packs_current_2page_2tok_v4_fp16 ^
  --sparse-hot-pack-device GPU ^
  --sparse-precompile-static ^
  --max-new-tokens 2 ^
  --cache-dir openvino_cache_controlled/full_gpu_v4
```

本机结果：首次写入缓存约 `7.46 GB`；同一配置热启动编译从约 `72.5 s` 降到约 `20.5 s`。清理受控缓存：

```shell
python -m openvino_adapt.manage_openvino_cache ^
  --cache-dir openvino_cache_controlled/full_gpu_v4 ^
  --max-gb 12
```

如果 Windows 暂时锁住 OpenVINO `.blob` 文件，工具会跳过并报告，稍后重试即可。

长输出路由统计：

```shell
python -m openvino_adapt.benchmark_openvino ^
  --decoder sparse ^
  --device GPU ^
  --image outputs_openvino_2page_input/page_0001.png outputs_openvino_2page_input/page_0002.png ^
  --prefill-model openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml ^
  --sparse-artifact-dir openvino_models/sparse_decode_past677_mixed_fp4 ^
  --sparse-device GPU ^
  --sparse-expert-device GPU ^
  --sparse-hot-pack-dir openvino_models/hot_expert_packs_current_2page_2tok_v4_fp16 ^
  --sparse-hot-pack-device GPU ^
  --sparse-precompile-static ^
  --max-new-tokens 12 ^
  --eos-token-id -1 ^
  --cache-dir openvino_cache_controlled/full_gpu_v4 ^
  --output-json outputs_openvino_sparse_logs/benchmark_full_gpu_v4_cache_profile_12tok.json
```

再生成长输出热包：

```shell
python -m openvino_adapt.make_hot_pack_plan_from_benchmark ^
  outputs_openvino_sparse_logs/benchmark_full_gpu_v4_cache_profile_12tok.json ^
  --max-experts 48 ^
  --output-json outputs_openvino_sparse_logs/hot_pack_plan_2page_12tok_top48.json

python -m openvino_adapt.export_hot_expert_packs_all ^
  --plan-json outputs_openvino_sparse_logs/hot_pack_plan_2page_12tok_top48.json ^
  --output-dir openvino_models/hot_expert_packs_2page_12tok_top48_fp16 ^
  --fp16
```

本机结果：v4 热包在 12-token 统计里有 `610` 次回退，解码循环约 `38.28 s`；top48 长输出热包降到 `3` 次回退，解码循环约 `2.13 s`，稀疏解码约 `5.19 tok/s`。

继续加入 GPU 路由边界漂移的安全余量后，生成 top48 v2 热包：

```shell
python -m openvino_adapt.export_hot_expert_packs_all ^
  --plan-json outputs_openvino_sparse_logs/hot_pack_plan_2page_12tok_top48_v2.json ^
  --output-dir openvino_models/hot_expert_packs_2page_12tok_top48_v2_fp16 ^
  --fp16
```

本机结果：

- 产物约 `2.58 GB`；
- 12-token 基准测试：`0` 次回退，解码循环约 `2.09 s`，稀疏解码约 `5.29 tok/s`；
- 热启动 12-token 基准测试：`0` 次回退，解码循环约 `2.19 s`，稀疏解码约 `5.06 tok/s`；
- 受控缓存当前约 `55.90 GB`，会随新图编译继续增长。

32-token 长输出会暴露更多冷门路由。继续用基准测试日志迭代热包：

```shell
python -m openvino_adapt.make_hot_pack_plan_from_benchmark ^
  outputs_openvino_sparse_logs/benchmark_full_gpu_top48_v2_cache_12tok.json ^
  outputs_openvino_sparse_logs/benchmark_full_gpu_top48_v2_cache_32tok.json ^
  outputs_openvino_sparse_logs/benchmark_full_gpu_top56_cache_32tok.json ^
  outputs_openvino_sparse_logs/benchmark_full_gpu_top60_cache_32tok.json ^
  --max-experts 61 ^
  --output-json outputs_openvino_sparse_logs/hot_pack_plan_2page_32tok_top61.json

python -m openvino_adapt.export_hot_expert_packs_all ^
  --plan-json outputs_openvino_sparse_logs/hot_pack_plan_2page_32tok_top61.json ^
  --output-dir openvino_models/hot_expert_packs_2page_32tok_top61_gather_fp16 ^
  --fp16 ^
  --gather

python -m openvino_adapt.export_sparse_decode_all ^
  --output-dir openvino_models/fused_hot_gather_past677_top61_fp16 ^
  --past-len 677 ^
  --layer-fp16 ^
  --fused-hot-gather ^
  --hot-plan-json outputs_openvino_sparse_logs/hot_pack_plan_2page_32tok_top61.json
```

本机 32-token 结果：

- top48 v2：`381` 次回退，解码循环约 `28.78 s`，整体约 `0.91 tok/s`；
- top61 dense-pack：`0` 次回退，解码循环约 `6.13 s`，稀疏解码约 `0.197 s/step`，整体约 `2.65 tok/s`；
- top61 gather-pack：`0` 次回退，解码循环约 `4.32 s`，稀疏解码约 `0.138 s/step`，整体约 `3.10 tok/s`；
- fused-hot-gather：`0` 次回退，解码循环约 `3.62 s`，稀疏解码约 `0.116 s/step`，整体约 `3.51 tok/s`；
- fused-hot-gather 产物约 `4.89 GB`。

### 多层 decoder block 融合

可以继续把多个 decoder layer 融成一个 OpenVINO 图：

```shell
python -m openvino_adapt.export_fused_decode_blocks ^
  --model models/Unlimited-OCR ^
  --output-dir openvino_models/fused_hot_gather_blocks4_past677_top61_fp16 ^
  --past-len 677 ^
  --hot-plan-json outputs_openvino_sparse_logs/hot_pack_plan_2page_32tok_top61.json ^
  --block-size 4 ^
  --fp16
```

运行时会自动识别 `metadata.json` 里的 `blocks`，走 block 图而不是逐层图。本机实测结论：

- block2：32-token 基准测试约 `3.11 tok/s`；
- block3：常驻服务第二任务约 `4.54 tok/s`；
- block4：常驻服务第二任务约 `4.59 tok/s`；
- block6：第一任务解码循环较快，但第二任务触发 Intel GPU `CL_OUT_OF_RESOURCES`，不稳定。

所以当前推荐仍然是 `fused_hot_gather_past677_top61_fp16` 这个单层 fused-hot-gather 产物。block 融合代码已经可用，但在这张 GPU 上，大图的寄存器/显存压力抵消了减少图调用的收益。

### final head 和短窗口实验

`final_norm_topkK.xml` 和 `final_norm_argmax.xml` 已经支持实验性导出和运行：

```shell
python -m openvino_adapt.export_sparse_decode_all ^
  --model models/Unlimited-OCR ^
  --output-dir openvino_models/fused_hot_gather_past677_top61_fp16 ^
  --past-len 677 ^
  --final-only ^
  --final-topk 16

python -m openvino_adapt.benchmark_openvino ^
  --decoder sparse ^
  --sparse-artifact-dir openvino_models/fused_hot_gather_past677_top61_fp16 ^
  --sparse-final-topk 16
```

本机结论：不推荐作为当前提速路径。`TopK(16)` 和 `ArgMax` 每步约 `18-20 ms`，反而慢于完整 logits final head 的约 `6-7 ms`。OpenVINO GPU 上排序/归约开销抵消了减少回传数据的收益。

短 KV 窗口也已测过：

- `ring_window=64 / past_len=613`：常驻服务第二个 32-token 任务约 `4.57 tok/s`；
- `ring_window=32 / past_len=581`：常驻服务第二个 32-token 任务约 `3.94 tok/s`。

所以当前仍推荐 `ring_window=128 / past_len=677`。

常驻 JSONL 服务可以把编译成本摊到连续任务：

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

服务启动后先输出 `ready` JSON 行。随后向标准输入写 JSONL 任务：

```json
{"id":"server-32tok-1","images":["outputs_openvino_2page_input/page_0001.png","outputs_openvino_2page_input/page_0002.png"],"max_new_tokens":32,"eos_token_id":-1}
```

同一服务进程内两次 12-token 任务的本机结果：

- 编译阶段约 `31.8 s`，只在服务启动时发生；
- 第一次任务总耗时约 `8.97 s`，稀疏解码约 `4.55 tok/s`；
- 第二次任务总耗时约 `5.45 s`，稀疏解码约 `7.04 tok/s`；
- 两次都是 `0` 次回退。

同一服务进程内两次 top61 gather 32-token 任务的本机结果：

- 编译阶段约 `30.6 s`，只在服务启动时发生；
- 第一次任务总耗时约 `9.98 s`，解码循环约 `4.01 s`；
- 第二次任务总耗时约 `7.93 s`，解码循环约 `3.88 s`，整体约 `4.05 tok/s`；
- 两次都是 `0` 次回退。

同一服务进程内两次 fused-hot-gather 32-token 任务的本机结果：

- 编译阶段约 `27.5 s`，只在服务启动时发生；
- 第一次任务总耗时约 `9.24 s`，解码循环约 `3.46 s`；
- 第二次任务总耗时约 `6.62 s`，解码循环约 `2.86 s`，整体约 `4.85 tok/s`；
- 两次都是 `0` 次回退。

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
- 全 GPU fused-hot-gather 稀疏解码是当前最快冒烟测试路径；
- GPU 路由存在少量边界漂移，hot pack 需要为漂移专家预留安全余量。

## OpenVINO 缓存

所有运行器支持：

```shell
--cache-dir openvino_cache
```

缓存和本机硬件、驱动、OpenVINO 版本相关，不应该提交到 Git。

## 已知限制

- 这是研究版适配，不是完整 OCR 产品。
- sparse decode 当前需要固定 shape。
- 不同页数、prompt、ring window 需要匹配导出。
- 本地 OpenVINO/NNCF 没有 INT2 weight compression。
- `experimental_int2` 只是未来自定义 kernel 的标记。
- PDF 运行器的 `manifest.json` 里页图路径可能是临时渲染路径。
- 稳定输出是 `continuous.md`、`page_*.md`、`combined.md` 和 `manifest.json`。

## 文件说明

- `export_openvino.py`：单组件导出。
- `export_all.py`：基础产物批量导出。
- `run_ocr_openvino.py`：图像/PDF OCR 运行器。
- `serve_ocr_openvino.py`：常驻 JSONL OCR 运行器，复用 OpenVINO 编译结果。
- `run_generate_openvino.py`：OpenVINO greedy generation loop。
- `run_sparse_decode_openvino.py`：完整 12 层 sparse decode runtime。
- `export_sparse_decode_all.py`：完整稀疏解码产物导出。
- `export_fused_decode_blocks.py`：多层 decoder block 融合产物导出。
- `export_hot_expert_packs_all.py`：按路由计划导出每层热专家包。
- `make_hot_pack_plan_from_benchmark.py`：从基准测试路由日志生成热包计划。
- `profile_sparse_experts.py`：专家路由统计。
- `make_expert_precision_plan.py`：生成 per-expert precision plan。
- `compress_sparse_experts.py`：按计划压缩 sparse experts。
- `manage_openvino_cache.py`：统计、裁剪或清理受控 OpenVINO 缓存。
- `benchmark_openvino.py`：端到端生成基准测试。
- `benchmark_decode_one.py`：解码图诊断基准测试。
- `summarize_benchmarks.py`：汇总基准测试 JSON 指标。
- `runtime.py`：OpenVINO 设备/缓存编译辅助。
