# 工控机专用屏幕部署

专用屏幕使用 Chromium kiosk 打开 `/display`。这个页面默认显示表情，不直接打开管理网页；网页模式由显示状态切换控制。

## Windows 本机启动

首次配置本机环境时，后台启动依赖和文档解析依赖需要一起安装。Word（DOCX）和 PowerPoint（PPTX）的自动解析使用 Docling，仅安装 `web,indexing` 可以打开网页，但无法解析这两种格式。在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[web,indexing,pdf,ocr,html,docling]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果解析任务提示缺少 `docling`，补装 `.[docling]` 后，在文档管理中重新解析已上传文件即可，无需重新上传。上述依赖不包含扫描件识别所需的全部本地模型，也不包含使用 Unstructured 的其他格式依赖。

在 Windows 文件管理器中双击项目根目录的 `一键启动小绿.bat`，即可启动服务、在电脑打开管理网页，并在 HDMI 扩展屏打开置顶的表情窗口。重复双击不会重复启动 Web 服务或表情窗口。

在项目根目录运行：

```powershell
.\scripts\start_windows.ps1
```

脚本会从项目虚拟环境启动 Web 服务，在电脑主屏幕的默认浏览器打开管理网页 `http://127.0.0.1:8000/`。如果 Windows 检测到非主屏幕，还会在外接屏打开全屏表情页 `http://127.0.0.1:8000/display`。外接屏应设为 Windows「扩展」模式；目前只有一块屏幕时，不会在电脑主屏幕自动打开表情页。外接屏后接入时，重新运行启动脚本即可；服务已经运行时不会重复启动。

HDMI 直连的屏幕也会被 Windows 识别。若系统能看到两台显示设备，但脚本只检测到一个桌面屏幕，通常处于「复制」模式；可在 Windows 显示设置中改为「扩展这些显示器」，或运行 `DisplaySwitch.exe /extend`，然后重新运行启动脚本。

在管理网页的「对话」中输入问题或使用麦克风时，外接屏会随状态切换：聆听、思考、回答或朗读、完成或错误，随后回到待机。表情页需要保持打开；切换到网页维护模式时，仍会记录表情状态，返回表情模式后继续显示。

### 网页与表情同步

网页为每次加载建立独立的显示会话，并每 2 秒同步当前状态。刷新会替换同一标签页的旧会话、恢复待机；旧页面的延迟请求或关闭通知不能覆盖新会话。其他闲置标签页不会抢占正在交互的页面，开始新提问或录音时才接管显示。网页维护模式保持不变。

页面正常关闭时发送释放通知。浏览器异常退出或网络中断时，最后一次有效同步约 10 秒后，服务端将该网页会话的表情恢复为待机；显示页自身连续约 3 秒无法读取状态时也会回到待机，并持续尝试重连。服务恢复后仍在运行的网页会重新建立同步。手动预览和 ROS2 状态不受网页心跳覆盖。

音频合成和缓冲期间显示思考，收到音频播放器的实际 `playing` 事件才显示说话；等待缓冲、暂停、结束、失败或打断时按剩余交互状态更新。旧朗读任务的清理回调不能清除新任务。完成/错误表情结束后重新检查录音、识别和回答状态，不会因一次计时器错过空闲时间而永久停留。

回归验证：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_display_sync.py tests/test_display_state.py tests/test_display_layout.py tests/test_web_app.py -k "display or voice or wakeup or tts" -q
playwright-cli -s=display-sync open http://127.0.0.1:8000/ --browser=msedge
playwright-cli -s=display-sync run-code --filename=tests/browser/display_sync.js
playwright-cli -s=display-sync run-code --filename=tests/browser/display_sync_recovery.js
playwright-cli -s=display-sync close
```

浏览器脚本使用受控回答流和本地 MP3 验证时序，不调用云端生成/合成接口。更新前已经打开的管理页和显示页需刷新一次，以加载新脚本。

若有多块外接屏，也可单独指定 Windows 屏幕编号（从 0 开始）：

```powershell
.\scripts\open_display_windows.ps1 -ScreenIndex 1
```

停止服务：

```powershell
.\scripts\stop_windows.ps1
```

本机不用 ROS2 也能切换显示模式或预览表情：

```powershell
.\.venv\Scripts\python.exe -m rag_system.display_state mode web
.\.venv\Scripts\python.exe -m rag_system.display_state mode emotion
.\.venv\Scripts\python.exe -m rag_system.display_state emotion thinking --text "正在思考"
.\.venv\Scripts\python.exe -m rag_system.display_state emotion idle
```

## 本地验证

启动 Web 服务后访问：

```text
http://127.0.0.1:8000/display
```

也可以通过 Web API 临时切换：

```bash
curl -X POST http://127.0.0.1:8000/api/display/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"web","reason":"maintenance"}'
```

## ROS2 控制

启动 `voice_rag.launch.py` 后，显示节点提供 `/display/set_mode`：

```bash
ros2 service call /display/set_mode rag_interfaces/srv/SetDisplayMode \
  "{mode: web, reason: 'ssh maintenance'}"

ros2 service call /display/set_mode rag_interfaces/srv/SetDisplayMode \
  "{mode: emotion, reason: 'return to robot display'}"
```

显示节点订阅 `/voice/display_text`，将已有的 `[query]`、`[delta]`、`[answer]`、`[error]` 等事件映射到表情状态。显示模式为 `web` 时仍会记录机器人状态，但不会覆盖网页。

## Chromium kiosk

建议使用独立 systemd 用户服务启动浏览器：

```ini
[Unit]
After=graphical-session.target

[Service]
Environment=DISPLAY=:0
ExecStart=/usr/bin/chromium --kiosk --app=http://127.0.0.1:8000/display --noerrdialogs --disable-session-crashed-bubble
Restart=always

[Install]
WantedBy=graphical-session.target
```

工控机重启后，`DisplayNode` 会把状态初始化为 `emotion + idle`，所以不会因为上次维护停留在网页模式。
