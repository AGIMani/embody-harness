# embody-harness

## VR Teleop 启动顺序

`add_scene_glb.py` 可以直接把 Quest/OpenXR 遥操作接入当前 Genesis 场景里的 Nero 机械臂、connector、L10 手和腕部 D405 相机。启动时需要先让 CloudXR/OpenXR runtime 保持运行，然后再启动场景。

### 1. 启动 CloudXR runtime

打开第一个终端，保持不要关闭：

```bash
conda activate genesis
python -m isaacteleop.cloudxr --accept-eula
```

如果这是第一次使用，按照终端提示接受 EULA 和证书。CloudXR 会生成并维护类似下面的环境文件：

```bash
~/.cloudxr/run/cloudxr.env
```

`add_scene_glb.py` 会自动读取这个文件，一般不需要手动 `source`。

### 2. 在 Quest 里进入 WebXR/CloudXR 会话

在 Quest 浏览器里打开 CloudXR/WebXR client 页面，通常是：

```text
https://<你的电脑IP>:48322
```

如果浏览器提示证书不安全，需要先接受证书。进入页面后启动 immersive VR session。只有头显真正进入 VR session 后，OpenXR 才能拿到可用的 XR system。

### 3. 启动 Genesis 场景并接入 VR

打开第二个终端：

```bash
cd /home/whf/Project/harness
conda activate genesis
python add_scene_glb.py --backend gpu --enable-vr-teleop --vr-markers-only
```

`--vr-markers-only` 用于先验证 VR 输入链路，不做 IK、不移动机械臂。确认日志里出现类似下面的信息后，再去掉这个参数：

```text
[quest-session] started
[add-scene-vr] Quest teleop is driving the add_scene_glb assembly
```

真正驱动当前 `add_scene_glb.py` 里的机械臂：

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop
```

控制左臂：

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop --vr-arm-side left
```

如果你需要更多时间戴上头显并进入 WebXR，会话等待时间可以拉长：

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop --vr-markers-only --vr-startup-timeout-s 600
```

### 常见报错

`Environment variable NV_CXR_RUNTIME_DIR is not set`：CloudXR 环境变量没有加载。通常重新运行 CloudXR runtime 即可；也可以手动执行：

```bash
source ~/.cloudxr/run/cloudxr.env
```

`Failed to connect to socket ... ipc_cloudxr` 或 `Connection refused`：CloudXR runtime 没有运行，或者 socket 是旧残留。重新启动第一个终端里的 CloudXR：

```bash
conda activate genesis
python -m isaacteleop.cloudxr --accept-eula
```

`Failed to get OpenXR system: -35`：CloudXR runtime 已经起来了，但 Quest/WebXR 客户端还没有真正进入 immersive VR session。保持 CloudXR 终端开着，在 Quest 里打开 `https://<你的电脑IP>:48322`，接受证书并进入 VR session，然后重新运行 `add_scene_glb.py` 或等待它重试。
