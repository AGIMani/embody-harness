# embody-harness

## VR 启动顺序

每个终端都先执行：

```bash
cd /home/whf/Project/harness
conda activate genesis
```

### 1. 启动 CloudXR runtime

```bash
python -m isaacteleop.cloudxr --accept-eula
```

### 2. 启动 CloudXR 网页端

```bash
scripts/run_cloudxr_web_client.sh
```

这个脚本启动的是完整网页：`https://192.168.8.100:8443/`。

`https://192.168.8.100:48322/` 只用于接受 CloudXR WSS 证书，不是操作页面，所以这里不会有 `Enable Voice`。

### 3. 启动语音桥

```bash
scripts/run_quest_voice_command_bridge.sh --model-path /home/whf/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22 --no-tls --min-confidence 0.5
```

### 4. Quest 打开网页

在 Quest 浏览器里按顺序打开：

```text
https://192.168.8.100:8443/
https://192.168.8.100:48322/
https://192.168.8.100:8443/
```

回到 `8443` 页面后，点击 `Enable Voice`，允许麦克风，然后点击 `Connect` 进入 immersive VR session。

### 5. 启动 VR 画面和手骨架输出

```bash
scripts/run_add_scene_vr_output.sh --display :0
```

### 6. 启动 Genesis 场景和遥操

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop
```

### 7. 开始遥操

进入 VR 后放下手柄，切到裸手追踪，然后说：

```text
开始
```

也可以用本机命令直接打开遥操：

```bash
scripts/send_teleop_voice_command_once.sh --command engage
```

### 8. 停止

按这个顺序 `Ctrl+C`：

```text
1. add_scene_glb.py
2. scripts/run_add_scene_vr_output.sh
3. scripts/run_quest_voice_command_bridge.sh
4. scripts/run_cloudxr_web_client.sh
5. python -m isaacteleop.cloudxr --accept-eula
```
