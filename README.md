# embody-harness

## VR 启动顺序

按下面顺序从上到下启动。

### 0. 每个终端先进入环境

```bash
cd /home/whf/Project/harness
conda activate genesis
```

### 1. 启动 CloudXR

开第一个终端，运行后保持不关：

```bash
python -m isaacteleop.cloudxr --accept-eula
```

### 2. Quest 进入 VR

在 Quest 浏览器打开：

```text
https://<你的电脑IP>:48322
```

接受证书，然后进入 immersive VR session。

进入后把两个 Quest 手柄放到一边，等 Quest 切到裸手追踪。

### 3. 启动 VR 画面输出

确认 Quest 已经停留在 immersive VR session 里，再开第二个终端：

```bash
cd /home/whf/Project/harness
conda activate genesis
scripts/run_add_scene_vr_output.sh --display :0
```

这一步会把电脑上的 Genesis 画面送到 Quest 里，并显示手骨架。

### 4. 启动 Genesis 场景和 VR 控制

开第三个终端：

```bash
cd /home/whf/Project/harness
conda activate genesis
python add_scene_glb.py --backend gpu --enable-vr-teleop
```

### 5. 停止

按这个顺序 `Ctrl+C`：

```text
1. add_scene_glb.py
2. scripts/run_add_scene_vr_output.sh
3. python -m isaacteleop.cloudxr --accept-eula
```
