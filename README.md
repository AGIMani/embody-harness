# embody-harness

## VR 启动

终端 1：

sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2
ls -l /dev/video44

```bash
cd /home/whf/Project/harness
conda activate genesis
scripts/run_add_scene_vr_prereqs.sh --display :0
```

这个脚本会同时启动 CloudXR runtime、CloudXR 网页端、语音桥、VR 画面和手骨架输出。
如果想把 Genesis 场景也放在同一个终端里一起启动，用：

```bash
scripts/run_add_scene_vr_prereqs.sh --display :0 --with-scene
```

在 Quest 浏览器里按顺序打开：

```text
https://192.168.8.100:8443/
https://192.168.8.100:48322/
https://192.168.8.100:8443/
```

回到 `8443` 页面后，点击 `Enable Voice`，允许麦克风，然后点击 `Connect` 进入 immersive VR session。

不用 `--with-scene` 时，再开终端 2：

```bash
cd /home/whf/Project/harness
conda activate genesis
python add_scene_glb.py --backend gpu --enable-vr-teleop
```

进入 VR 后放下手柄，切到裸手追踪，然后说：

```text
开始
```

也可以用本机命令直接打开遥操：

```bash
scripts/send_teleop_voice_command_once.sh --command engage
```

停止时先在终端 2 按 `Ctrl+C` 停 Genesis，再在终端 1 按 `Ctrl+C` 停所有 VR 前置服务。
