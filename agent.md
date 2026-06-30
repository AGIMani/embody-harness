# GR00T Finetune CKPT 模型接口说明

本文档说明当前 `checkpoints/finetune` 内 GR00T finetune checkpoint 的推理接口：需要哪些基模和 checkpoint 文件、模型输入怎么准备、模型输出是什么、输出如何转成机器人控制目标。这里重点是模型 I/O 协议，不是某个脚本的使用说明。

## 1. 需要准备的模型和代码位置

### 1.1 Isaac-GR00T 源码

推理需要 Isaac-GR00T Python 包注册 `Gr00tN1d7` 模型和 processor。当前默认源码位置：

```text
/home/whf/Project/Isaac-GR00T
```

也可以通过环境变量覆盖：

```bash
export ISAAC_GROOT_ROOT=/path/to/Isaac-GR00T
```

推理前需要把该路径加入 `sys.path`，并 import `gr00t.model`，否则 `transformers.AutoModel/AutoProcessor` 找不到自定义 GR00T 类。

### 1.2 Finetune checkpoint

当前可用 finetune checkpoint 目录包括：

```text
checkpoints/finetune/checkpoint-20000
checkpoints/finetune/checkpoint-59000
```

一个可推理 checkpoint 至少需要：

- `config.json`: GR00T 模型结构配置，其中 `model_name` 指向 Cosmos 基模。
- `processor_config.json`: observation/action modality 配置和 processor 参数。
- `statistics.json`: processor 做 state/action 归一化和反归一化时读取的统计量。
- `embodiment_id.json`: embodiment 到 id 的映射，如存在则会被 processor 读取。
- `model.safetensors.index.json` 和 `model-*.safetensors`: finetune 后的模型权重。

`experiment_cfg/dataset_statistics.json` 是训练记录备份；推理 processor 默认读取 checkpoint 根目录的 `statistics.json`。

### 1.3 Cosmos 基模

当前 finetune ckpt 的 VLM/backbone 基模是：

```text
checkpoints/nvidia/Cosmos-Reason2-2B
```

该目录需要包含 tokenizer、processor 和基模权重，例如：

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
preprocessor_config.json
video_preprocessor_config.json
chat_template.json
```

注意：`checkpoint-20000/config.json` 里的 `model_name` 可能还是训练机器上的绝对路径，例如 `/root/autodl-tmp/.../Cosmos-Reason2-2B`。本机推理时要把 model config 和 processor kwargs 的 `model_name` 覆盖成本机 Cosmos 路径，或者临时复制/patched checkpoint 配置后再 `from_pretrained`。

## 2. 当前 embodiment 和 modality schema

当前 finetune 使用 `EmbodimentTag.NEW_EMBODIMENT`，也就是 processor 配置里的 `new_embodiment`。

### 2.1 Video 输入

```text
video.delta_indices = [0]
video.modality_keys = ["ego_view", "wrist_view"]
```

需要同时提供两路 RGB 图像：

- `ego_view`: 外部/头部视角，当前场景中对应 D455 RGB。
- `wrist_view`: 右腕视角，当前场景中对应右手 D405。

每个 key 的数组 shape：

```text
(B, T, H, W, C) = (1, 1, 180, 320, 3)
dtype = uint8
color = RGB
```

`T=1` 来自 `delta_indices=[0]`，只用当前帧。图像不要传 BGR；如果来自 OpenCV，需要先 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`。当前默认输入尺寸是 `height,width = 180,320`。

### 2.2 State 输入

```text
state.delta_indices = [0]
state.modality_keys = ["eef_9d", "hand_joint_pos", "arm_joint_pos"]
state.sin_cos_embedding_keys = ["arm_joint_pos", "hand_joint_pos"]
```

每个 state key 的数组 shape 都是：

```text
(B, T, D) = (1, 1, D)
dtype = float32
```

字段含义：

- `eef_9d`: 右臂末端位姿，9D，`[x, y, z, rot6d_0..rot6d_5]`。
- `hand_joint_pos`: Linker L10 右手 proprioception，10D，使用 GR00T/L10 canonical joint 顺序。
- `arm_joint_pos`: Nero 右臂关节角，7D，单位 rad。

`eef_9d` 的 rot6d 使用旋转矩阵前两列拼接的 6D 表示。保持和训练数据一致最重要：xyz 应该在 policy/robot-base/action frame 中，rot6d 也应是同一坐标语义下的末端姿态。

L10 canonical 10D 手指顺序：

```text
thumb_cmc_pitch
thumb_cmc_yaw
index_mcp_pitch
middle_mcp_pitch
ring_mcp_pitch
pinky_mcp_pitch
index_mcp_roll
ring_mcp_roll
pinky_mcp_roll
thumb_cmc_roll
```

### 2.3 Language 输入

```text
language.delta_indices = [0]
language.modality_keys = ["annotation.human.action.task_description"]
```

语言输入使用嵌套 list：

```python
language = {
    "annotation.human.action.task_description": [[instruction]],
}
```

当前默认任务语义是“拿起绿盖瓶子并放到白色矩形区域”。实际部署时可以替换为当前任务描述，但要尽量贴近训练分布。

## 3. Observation 组织格式

最终传给 policy 的 observation 是三层 dict：

```python
observation = {
    "video": {
        "ego_view": ego_rgb[None, None, ...].astype(np.uint8),
        "wrist_view": wrist_rgb[None, None, ...].astype(np.uint8),
    },
    "state": {
        "eef_9d": eef_9d[None, None, ...].astype(np.float32),
        "hand_joint_pos": hand_joint_pos[None, None, ...].astype(np.float32),
        "arm_joint_pos": arm_joint_pos[None, None, ...].astype(np.float32),
    },
    "language": {
        "annotation.human.action.task_description": [[instruction]],
    },
}
```

最容易出错的点：

- 图像 shape 是 `B,T,H,W,C`，不是 `B,C,H,W`。
- 图像 dtype 是 `uint8` RGB，不要提前归一化到 `[0,1]`。
- state/action 归一化由 checkpoint 的 `statistics.json` 和 processor 完成，外部不要手动 normalize。
- `hand_joint_pos` 是训练采集链路里的 L10 reported/proprioception command-space 数值，不一定等于直接写入仿真的 URDF raw qpos。
- `arm_joint_pos` 和 `hand_joint_pos` 会做 sin/cos state encoding，数值单位和顺序必须稳定。

## 4. 推理调用流程

推荐流程是：

```python
import sys
from pathlib import Path
import torch

ISAAC_GROOT_ROOT = Path("/home/whf/Project/Isaac-GR00T")
sys.path.insert(0, str(ISAAC_GROOT_ROOT))

import gr00t.model  # registers AutoModel/AutoProcessor classes
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from transformers import AutoConfig, AutoModel, AutoProcessor

checkpoint = Path("checkpoints/finetune/checkpoint-20000").resolve()
cosmos_model = Path("checkpoints/nvidia/Cosmos-Reason2-2B").resolve()

config = AutoConfig.from_pretrained(checkpoint)
config.model_name = str(cosmos_model)

model = AutoModel.from_pretrained(
    checkpoint,
    config=config,
    local_files_only=True,
    transformers_loading_kwargs={"local_files_only": True},
).eval().to(device="cuda:0", dtype=torch.bfloat16)

processor = AutoProcessor.from_pretrained(
    checkpoint,
    model_name=str(cosmos_model),
    transformers_loading_kwargs={"local_files_only": True},
).eval()

embodiment = EmbodimentTag.NEW_EMBODIMENT
modality = processor.get_modality_configs()[embodiment.value]
language_key = modality["language"].modality_keys[0]

single_obs = {
    "video": {key: value[0] for key, value in observation["video"].items()},
    "state": {key: value[0] for key, value in observation["state"].items()},
    "language": {key: value[0] for key, value in observation["language"].items()},
}

vla_step = VLAStepData(
    images=single_obs["video"],
    states=single_obs["state"],
    actions={},
    text=single_obs["language"][language_key][0],
    embodiment=embodiment,
)
messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]

processed = processor(messages)
collated = processor.collator([processed])
collated = move_to_device_and_bf16(collated, "cuda:0")

with torch.inference_mode():
    pred = model.get_action(**collated)

normalized_action = pred["action_pred"].float().cpu().numpy()
batched_states = {
    key: observation["state"][key]
    for key in modality["state"].modality_keys
}
decoded_action = processor.decode_action(normalized_action, embodiment, batched_states)
```

上面 `move_to_device_and_bf16` 需要递归处理 tensor/list/dict。实际使用时也可以直接用 `Gr00tPolicy.get_action(observation)`，但要确保 Cosmos 路径和 processor 路径都能被正确解析。

## 5. 输出 action 的结构和含义

当前 action schema：

```text
action.delta_indices = [0, 1, ..., 31]
action.modality_keys = ["eef_9d", "hand_joint_target", "arm_joint_target"]
```

`processor.decode_action(...)` 后返回 unnormalized action dict，常见 shape：

```text
decoded_action["eef_9d"]            -> (B, 32, 9), float32
decoded_action["hand_joint_target"] -> (B, 32, 10), float32
decoded_action["arm_joint_target"]  -> (B, 32, 7), float32
```

如果 batch size 是 1，执行时通常先取第一维，得到 `(32,D)` 的 action chunk。每个 chunk 是未来 32 个 action step。当前控制通常每次重规划后只执行前 `replan_horizon=8` 步，然后重新观测和推理。

### 5.1 `eef_9d`

`eef_9d` 是右臂末端目标位姿：

```text
[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5]
```

训练配置里它是 `RELATIVE EEF XYZ_ROT6D`，但 `decode_action` 已经结合当前 `state["eef_9d"]` 反解回可执行的目标位姿。因此对接控制器时把 decoded `eef_9d[t]` 当作 policy/action frame 下的目标末端位姿，而不是再手动加一次当前 state。

使用方式：

1. 取当前执行步 `eef = decoded_action["eef_9d"][0, action_index]`。
2. 将 `eef[:3]` 作为目标 xyz。
3. 将 `eef[3:9]` 从 rot6d 转回 3x3 rotation matrix。
4. 如果控制器在 world frame 下工作，先把 policy/action frame pose 转换到机器人/world frame。
5. 用 IK 求右臂 7D 关节命令，或者直接给支持 task-space 的控制器。

### 5.2 `hand_joint_target`

`hand_joint_target` 是 Linker L10 右手 10D canonical command target，顺序和 `hand_joint_pos` 一致。训练配置里它是 relative non-EEF，但 `decode_action` 后已经是目标 command 数值。

使用方式：

1. 取 `hand = decoded_action["hand_joint_target"][0, action_index, :10]`。
2. clip 到 policy command range，当前通用范围是 `[-0.6, 1.6]`。
3. 对四个非拇指 pitch 做非负保护：`index/middle/ring/pinky_mcp_pitch >= 0.0`。
4. 如需防抖或安全，可对上一帧命令做 rate limit。
5. 按 L10 canonical 顺序映射到实际手控制接口。

### 5.3 `arm_joint_target`

`arm_joint_target` 是 7D 右臂关节目标。当前 Nero export 中原始 19D action 没有显式 arm joint target；训练 metadata 将它 alias 到未来 recorded `arm_joint_pos`，主要用于保持接口完整和调试。

实际控制中优先使用 `eef_9d` 做末端 IK。只有当没有 `eef_9d` 或专门想测试 joint-space 输出时，才使用 `arm_joint_target`。

## 6. action_configs 对 decode 的影响

当前 `new_embodiment` 的 action config 是：

```text
eef_9d:            RELATIVE, EEF,     XYZ_ROT6D, state_key=eef_9d
hand_joint_target: RELATIVE, NON_EEF, DEFAULT,   state_key=hand_joint_pos
arm_joint_target:  RELATIVE, NON_EEF, DEFAULT,   state_key=arm_joint_pos
```

这意味着模型头输出的是 normalized action；processor 会根据 `statistics.json`、当前 state 和每个 action key 的 `state_key` 做反归一化和 relative-to-absolute 转换。外部使用 decoded action 时不要再按 relative action 逻辑二次叠加。

## 7. 最小执行循环

典型在线控制循环：

```text
1. 采集 ego_view/wrist_view RGB 当前帧，resize 到 180x320。
2. 读取当前右臂 q、右手 reported q、右臂 EEF pose。
3. 组 observation = video + state + language。
4. 调用 model.get_action，再用 processor.decode_action 得到 action chunk。
5. 执行 action chunk 的前 N 步，常用 N=8：
   - eef_9d -> policy pose -> world pose -> IK -> arm command
   - hand_joint_target -> clip/rate limit -> L10 hand command
6. 重新采集 observation，再 replan。
```

如果启用 RTC/previous action seed window，上一轮 decoded absolute action chunk 会作为 `VLAStepData.actions` 传给 processor/model，用于连续 chunk 的平滑衔接。没有 RTC 时 `actions={}` 即可。

## 8. 坐标和数值约定

- `eef_9d` 输入和输出必须使用同一个 policy/action frame 语义。当前 harness 会把 Genesis world EEF pose 转到 policy frame 后喂模型，再把 decoded policy pose 转回 world pose 执行。
- rot6d 必须能恢复成合法旋转矩阵；常用做法是对两个 3D 向量 Gram-Schmidt 正交化。
- `hand_joint_pos` 使用 L10 SDK reported/proprioception command-space 数值；如果实际硬件回读是 raw 0-255 或 URDF qpos，需要先转换到训练时的 command-space。
- `hand_joint_target` 输出保持在 policy command space；不要再投影到另一套 URDF lower/upper 后反馈给下一帧 observation，否则容易破坏 action-relative 链路。
- state/action 的统计范围在 `statistics.json` 中，`use_percentiles=true`、`use_relative_action=true`，所以异常离群输入会显著影响 decode 质量。

## 9. 快速自检

推理前检查：

- `checkpoint/config.json` 和 `checkpoint/processor_config.json` 的 `model_name` 能指到本机 `Cosmos-Reason2-2B`。
- checkpoint 根目录有 `statistics.json`。
- `processor.get_modality_configs()["new_embodiment"]` 中 video keys 是 `ego_view,wrist_view`。
- 第一帧 observation 中 `ego_view/wrist_view` shape 是 `(1,1,180,320,3)`，dtype 是 `uint8`。
- `eef_9d/hand_joint_pos/arm_joint_pos` shape 分别是 `(1,1,9)`、`(1,1,10)`、`(1,1,7)`。
- decode 后 action keys 至少包含 `eef_9d` 和 `hand_joint_target`，shape 分别是 `(1,32,9)` 和 `(1,32,10)`。
