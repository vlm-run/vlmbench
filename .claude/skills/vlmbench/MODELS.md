# Supported Models

Default vLLM serving arguments for OCR/VLM models. Use `--serve-args` to pass these when running benchmarks:

```bash
vlmbench run --serve --backend vllm --model <model-id> --serve-args '<args>' -i <input>
```

| Model | Params | vLLM `--serve-args` | Notes |
|---|---|---|---|
| `lightonai/LightOnOCR-2-1B` | 1B | `--limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching` | |
| `zai-org/GLM-OCR` | 0.9B | `--allowed-local-media-path /` | Requires transformers >= 5.0.0 |
| `rednote-hilab/dots.ocr` | 3B | `--trust-remote-code --gpu-memory-utilization 0.95` | |
| `allenai/olmOCR-2-7B-1025-FP8` | 8B (FP8) | `--max-model-len 16384` | Based on Qwen2.5-VL-7B |
| `Qwen/Qwen3-VL-8B-Instruct` | 9B | `--mm-encoder-tp-mode "data"` | |
| `Qwen/Qwen3-VL-8B-Instruct-FP8` | 9B (FP8) | `--mm-encoder-tp-mode "data"` | |
| `deepseek-ai/DeepSeek-OCR-2` | 3B | N/A | Not supported in upstream vLLM; requires [custom wheel](https://github.com/deepseek-ai/DeepSeek-OCR-2) |
