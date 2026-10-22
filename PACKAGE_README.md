# 绿色江河项目（脱敏分发包）

本压缩包包含项目源码、测试、文档、样例、ROS 2 接口与已解析文档。

当前默认语音流程已改为云端流式 ASR、EmbeddingGemma 2 实时检索和 Qwen Audio 3.1 原生音频回复。安装、旧库迁移、新 ROS2 入口与验证结果见 [直接语音回复流程](docs/direct_voice_pipeline.md)；旧语音节点和分模块示例保留用于兼容。

本机合并版已加入 3.0 的八种动态内容表情、预览页面和约 100 字回答要求；3.0 独立保留。
Linux 默认在 `http://127.0.0.1:8001` 运行，向量服务在 `127.0.0.1:8765`。
安装、启动、停止和验证说明见 [4.0 合并版运行说明](docs/merged_runtime.md)。

## 已包含

- `src/`、`tests/`、`docs/`、`scripts/`、`ros2/`
- `samples/` 中的公开/受控样例文件
- `parsed_documents/` 中的解析 JSON。来源字段已改写为包内相对路径，去除了本机用户名、微信目录和上传目录信息。
- `data/settings.example.json`：不含密钥的配置示例
- `docs/robot_deployment.md`：Ubuntu 22.04 + ROS 2 Humble 机器人端部署、启动、验证与故障排查指南

## 已排除

- Python/ROS 环境和模型权重：`.venv/`、`.venv-linux/`、`data/models/`、`data/tessdata/`
- 运行时数据、上传原文件、索引数据库、图片/音频缓存、日志、构建产物与临时文件
- `data/settings.json` 中的 API Key，以及仓库元数据和本机代理配置

解压后按 `AGENTS.md`/项目文档安装依赖；API Key 请通过环境变量或本地配置界面提供，不要写入版本库。
