# CosyVoice-25hz
 项目原地址：https://github.com/FunAudioLLM/CosyVoice.git
 刘悦的技术博客整合版原地址：https://github.com/v3ucn/CosyVoice_For_Windows.git
 learnWave项目地址： 后端 https://gitee.com/sgner/LearnWave-EduSynth-System.git 前端 https://gitee.com/sgner/learn-wave.git
# API 文档 - 音频生成接口

本文档为 CosyVoice 项目封装的音频生成 API，通过指定不同的模式（mode）来实现多种语音合成功能。该接口利用阿里云 OSS 实现用户资源的下载和上传，支持流式传输和非流式输出。

## 音频生成接口
 ## all_api.py
- **端点**: `/generate_audio`
- **方法**: POST
- **描述**: 根据指定模式生成音频，支持预训练音色、3秒极速复刻、跨语种复刻和自然语言控制四种模式。接口通过阿里云 OSS 下载或上传用户相关资源（如模型权重和参考音频），并根据 `stream` 参数决定是否以流式方式返回音频数据。
- **请求体**:
  - `mode` (字符串, 必需): 音频生成模式，可选值：
    - `"预训练音色"`: 使用预训练音色生成语音。
    - `"3s极速复刻"`: 基于参考音频进行快速语音复刻。
    - `"跨语种复刻"`: 实现跨语言的语音复刻。
    - `"自然语言控制"`: 通过自然语言指令控制语音生成。
  - `tts_text` (字符串, 必需): 要转换为语音的文本内容。
  - `spk_id` (字符串, 必需): 预训练音色的标识符（仅在 `"预训练音色"` 和 `"自然语言控制"` 模式下使用）。
  - `spk_customize` (字符串, 可选, 默认="无"): 自定义音色标识符，若非 "无"，将尝试从阿里云 OSS 下载对应音色模型（路径: `learn-wave/{user_id}/model/cosyvoice/{spk_customize}.pt`）。
  - `prompt_text` (字符串, 可选): 提示文本，用于 `"3s极速复刻"` 和 `"自然语言控制"` 模式。
  - `prompt_speech_16k` (字符串, 可选): 参考音频路径（16kHz采样率），用于 `"3s极速复刻"` 和 `"跨语种复刻"` 模式，若本地不存在则从阿里云 OSS 下载。
  - `seed` (整数, 可选, 默认=0): 随机种子，用于确保生成结果的可重复性。
  - `stream` (字符串, 可选, 默认="True"): 是否启用流式传输，可选值 `"True"` 或 `"False"`。
  - `speed` (浮点数, 可选, 默认=1.0): 语音生成速度因子，调整播放速度。
  - `user_id` (字符串, 可选, 默认=""): 用户标识符，用于构造资源路径。
  - `exp_name` (字符串, 可选, 默认=""): 实验名称，用于 `"3s极速复刻"` 模式保存生成的模型文件。
- **响应**:
  - **流式传输 (`stream="True"`)**:
    - 返回类型: `audio/mpeg` 流式数据。
    - 头部:
      - `Content-Type`: `audio/mpeg`
      - `Content-Disposition`: `inline; filename=sound.mp3`
    - 内容: 逐步生成并返回 MP3 格式的音频流。
    - 清理: 处理完成后删除用户临时文件（路径: `{ROOT_DIR}/output/{user_id}` 或参考音频目录）。
  - **非流式传输 (`stream="False"`)**:
    - 返回类型: `audio/wav` 文件。
    - 头部: `mimetype="audio/wav"`
    - 内容: 一次性返回 WAV 格式的完整音频数据。
    - 清理: 处理完成后删除用户临时文件。
  - **特殊行为**:
    - 若 `mode="3s极速复刻"`，生成的模型文件会上传至阿里云 OSS（路径: `learn-wave/{user_id}/model/cosyvoice/{exp_name}.pt`）。
    - 若涉及参考音频，音频文件将从阿里云 OSS 下载至本地 `{ROOT_DIR}/参考音频/` 目录，处理完成后删除。
- **注意事项**:
  - 所有模式均依赖全局 `cosyvoice` 对象进行语音生成。
  - 音频采样率由全局变量 `target_sr` 和 `prompt_sr` 决定（未在参数中显式指定）。
  - 若资源下载或上传失败，可能抛出异常并记录错误日志。
