# Colab 运行说明

这个目录提供一套只放在 `colab/` 下的 Google Colab 运行文件，用来在 Colab GPU 上启动当前项目的同一套 `FastAPI` 服务。

包含文件：

- `qwen_asr_tts_api_colab.ipynb`: Colab notebook，负责安装依赖、下载模型、启动 API。
- `colab_runtime.py`: notebook 调用的辅助脚本，负责安装、模型下载、环境变量和 API 进程管理。
- `smoke_test.py`: Colab 侧冒烟测试工具，负责 GPU、`/healthz`、TTS 和可选 ASR 验证。
- `requirements-colab.txt`: Colab Python 依赖清单。

## 使用方式

1. 在 Colab 中打开 `colab/qwen_asr_tts_api_colab.ipynb`。
2. 连接 `GPU` 运行时。
3. 修改第一个配置单元：
   - 如果你会在 Colab 中 `git clone` 当前项目，填 `REPO_GIT_URL`。
   - 如果项目已经放在 Google Drive，改 `DRIVE_REPO_DIR`。
4. 依次运行 notebook 单元。

建议最少执行到这些验证单元：

- GPU 检查：确认 `nvidia-smi` 和 `torch.cuda.is_available()` 都正常。
- 服务健康检查：确认 `/healthz` 返回 `asr_ready=true`。
- TTS 冒烟测试：生成一个 `wav` 文件。
- 可选 ASR 冒烟测试：用仓库里的 `mnt_data/input.wav` 试一次转写。

## 说明

- notebook 会复用根目录已有的 `api_server.py`、`asr_engine.py`、`tts_engine.py`，不会改动根目录文件。
- 默认支持把模型下载到 Google Drive，这样下次启动可复用，不必重新下载。
- 可选开启 `Cloudflare Tunnel`，把 Colab 内部的 `8000` 端口暴露为公网地址。
- 如果你只需要 ASR，可以把配置里的 `ENABLE_TTS` 设为 `False`，减少模型下载和显存占用。
