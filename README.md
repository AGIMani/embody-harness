# embody-harness

## VR 启动顺序

每个终端先进入环境：

```bash
cd /home/whf/Project/harness
conda activate genesis
```

### 0. 首次准备

```bash
scripts/apply_isaac_teleop_cloudxr_overlay.sh
```

### 1. 启动 CloudXR

```bash
python -m isaacteleop.cloudxr --accept-eula
```

### 2. 启动 Quest 语音桥

```bash
scripts/run_quest_voice_command_bridge.sh --model-path /home/whf/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22 --no-tls --min-confidence 0.5
```

### 3. Quest 进入 immersive VR session

在 Quest 的 CloudXR WebXR 页面点 `Enable Voice`，允许麦克风，然后进入 immersive VR session。

如果只是接受证书页面，就先打开：

```text
https://<你的电脑IP>:48322
```

接受证书后回到 CloudXR WebXR 页面。进入 VR 后放下手柄，切到裸手追踪。

### 4. 启动 VR 画面和手骨架输出

```bash
scripts/run_add_scene_vr_output.sh --display :0
```

### 5. 启动 Genesis 场景和遥操

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop
```

启动后默认会等待“开始”命令，不会立刻跟手动。

### 6. 开始遥操

对 Quest 说：

```text
开始
```

也可以用本机命令验证链路：

```bash
scripts/send_teleop_voice_command_once.sh --command engage
```

之后可以用这些命令控制：

```bash
scripts/send_teleop_voice_command_once.sh --command clutch
scripts/send_teleop_voice_command_once.sh --command resume
scripts/send_teleop_voice_command_once.sh --command recenter
scripts/send_teleop_voice_command_once.sh --command stop
```

### 7. 停止

按这个顺序 `Ctrl+C`：

```text
1. add_scene_glb.py
2. scripts/run_add_scene_vr_output.sh
3. scripts/run_quest_voice_command_bridge.sh
4. python -m isaacteleop.cloudxr --accept-eula
```
