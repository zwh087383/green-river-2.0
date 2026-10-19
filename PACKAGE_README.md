# 绿色江河项目（脱敏分发包）

本压缩包包含项目源码、测试、文档、样例、ROS 2 接口与已解析文档。

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
