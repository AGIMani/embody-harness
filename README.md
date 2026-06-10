# embody-harness

## VR Teleop 启动顺序

`add_scene_glb.py` 现在分成两条链路：

- VR 输入：Quest/OpenXR 手势或控制器驱动当前 Genesis 场景里的 Nero 机械臂、connector、L10 手和腕部 D405。
- VR 输出：Genesis viewer 画面经 `ffmpeg` 抓屏写入 V4L2 loopback，再由 IsaacTeleop `camera_streamer` 在 Quest/CloudXR 里显示为 XR 平面，并默认开启手势追踪骨架 overlay。

### 0. 准备 sim-screen V4L2 设备

第一次使用前确认 loopback 设备存在：

```bash
ls -l /dev/video44
```

如果不存在，创建一次：

```bash
sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2
```

如果系统还没有 `v4l2loopback`，Ubuntu 22.04 + 6.8 HWE 内核不要优先用 apt 里的旧 `v4l2loopback-dkms 0.12.7`，它可能因为 `strlcpy` 编译失败。推荐用新版源码安装：

```bash
sudo apt install -y git build-essential dkms linux-headers-$(uname -r) v4l2loopback-utils
cd /tmp
rm -rf v4l2loopback
git clone https://github.com/v4l2loopback/v4l2loopback.git
cd v4l2loopback
make
sudo make install
sudo depmod -a
```

### 1. 启动 CloudXR runtime

打开第一个终端，保持不要关闭：

```bash
conda activate genesis
python -m isaacteleop.cloudxr --accept-eula
```

CloudXR 会生成环境文件：

```bash
~/.cloudxr/run/cloudxr.env
```

本仓库脚本会自动读取这个文件，一般不需要手动 `source`。

### 2. 在 Quest 里进入 WebXR/CloudXR 会话

在 Quest 浏览器里打开：

```text
https://<你的电脑IP>:48322
```

如果浏览器提示证书不安全，先接受证书。然后进入 immersive VR session。只有头显真正进入 VR session 后，OpenXR 才能拿到可用的 XR system。

### 3. 启动 VR 输出栈

打开第二个终端。先检查依赖和路径：

```bash
cd /home/whf/Project/harness
conda activate genesis
scripts/run_add_scene_vr_output.sh --display :0 --check-only
```

把 `--display :0` 改成 Genesis viewer 所在的 X11 display。如果你的终端里 `echo $DISPLAY` 有值，也可以省略 `--display`。

检查通过后启动输出栈：

```bash
scripts/run_add_scene_vr_output.sh --display :0
```

第一次启动时，如果本机还没有 `harness-camera-streamer-lite:latest`，脚本会自动构建一个从 `zhangbt@192.168.8.109:~/Teleop` 迁移来的 V4L2-only camera_streamer lite image。这条路线会生成临时 Dockerfile，跳过上游 `docker/dockerfile:1` BuildKit frontend 和 ZED SDK 安装，并把 3D hand skeleton overlay 编进 `xr_plane_renderer`。

也可以提前手动构建：

```bash
scripts/build_camera_streamer_lite.sh
```

默认行为：

- 抓取 `:0+0,0` 的 `1280x720@20fps` 画面。
- 写入 `/dev/video44`。
- 生成 IsaacTeleop `camera_streamer` XR 配置。
- 使用迁移来的 lite camera_streamer image，支持 producer-paced V4L2 loopback 和 hand skeleton overlay。
- 在 Quest 内显示仿真画面平面，并默认显示左右手骨架。

关闭手骨架 overlay 用：

```bash
scripts/run_add_scene_vr_output.sh --display :0 --disable-hand-overlay
```

常用画面参数：

```bash
scripts/run_add_scene_vr_output.sh \
  --display :0 \
  --device /dev/video44 \
  --size 1280x720 \
  --fps 20 \
  --plane-distance 1.6 \
  --plane-width 1.2
```

### 4. 启动 Genesis 场景并接入 VR 输入

打开第三个终端：

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

真正驱动当前场景里的机械臂：

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop
```

控制左臂：

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop --vr-arm-side left
```

如果需要更多时间戴上头显并进入 WebXR，会话等待时间可以拉长：

```bash
python add_scene_glb.py --backend gpu --enable-vr-teleop --vr-markers-only --vr-startup-timeout-s 600
```

## 常见报错

`Environment variable NV_CXR_RUNTIME_DIR is not set`：CloudXR 环境变量没有加载。通常重新运行 CloudXR runtime 即可；也可以手动执行：

```bash
source ~/.cloudxr/run/cloudxr.env
```

`Failed to connect to socket ... ipc_cloudxr` 或 `Connection refused`：CloudXR runtime 没有运行，或者 socket 是旧残留。重新启动第一个终端里的 CloudXR：

```bash
conda activate genesis
python -m isaacteleop.cloudxr --accept-eula
```

`Failed to get OpenXR system: -35`：CloudXR runtime 已经起来了，但 Quest/WebXR 客户端还没有真正进入 immersive VR session。保持 CloudXR 终端开着，在 Quest 里打开 `https://<你的电脑IP>:48322`，接受证书并进入 VR session，然后重新运行 VR 输出栈或 `add_scene_glb.py`。

`sim screen device missing: /dev/video44`：V4L2 loopback 设备不存在。运行：

```bash
sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2
```

`DISPLAY is empty`：当前 shell 不知道要抓哪个 X11 display。进入 Genesis viewer 所在桌面会话的终端运行 `echo $DISPLAY`，然后把结果传给 `--display`。

Quest 里有手势输入但没有骨架 overlay：先确认没有设置 `TELEOP_CAMERA_DISABLE_HAND_OVERLAY=1`，再看 VR 输出栈日志里是否出现 `hand skeleton overlay=enabled`。如果日志提示 `holohub.xr bindings do not expose XrHandTracker`，说明当前 IsaacTeleop/camera_streamer 构建不包含手追踪 overlay 支持，需要重新构建 camera_streamer XR image。

`docker/dockerfile:1 ... i/o timeout`：这是上游 `camera_streamer.sh build` 路线访问 Docker Hub 超时。默认 `scripts/run_add_scene_vr_output.sh` 已切到 lite image 路线，不再需要这一路。只有显式传 `--use-upstream-camera-streamer` 时才会走上游构建。
