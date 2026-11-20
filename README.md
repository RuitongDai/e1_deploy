# E1 Wheel 仿真与实机部署

本项目用同一套 ONNX 策略运行 MuJoCo 仿真和 E1 Wheel 真机。策略接收 69 维观测，输出 16 个腿关节位置目标和 4 个轮速目标。策略在 CPU 上运行，控制周期目标为 20 ms。

## 安装环境

在 Linux 的项目根目录执行。已有 `e1_deploy/` 时，可以直接激活并安装依赖。

```bash
uv venv --python 3.10 --prompt e1_deploy e1_deploy
source e1_deploy/bin/activate
uv pip install --python e1_deploy/bin/python -r requirements.txt
```

实机脚本从仓库的 `droidup/api/` 导入 gRPC 接口。所需的 `grpcio`、`protobuf` 和仿真依赖都在根目录的 `requirements.txt` 中。

## 运行

以下命令在项目根目录、激活 `e1_deploy` 后执行。

**验证模型或手柄**

```bash
python e1_wheel/scripts/sim2sim_e1_wheel.py --validate-policy-only
python e1_wheel/scripts/sim2sim_e1_wheel.py --test-gamepad
```

**MuJoCo 仿真**

```bash
python e1_wheel/scripts/sim2sim_e1_wheel.py
```

**真机**

先核对 [robot_config.json](e1_wheel/scripts/robot_config.json) 里的 SDK 地址、关节映射和控制模式。当前默认是前腿及前轮 `192.168.51.3:50054`，后腿及后轮 `192.168.51.3:50053`。

```bash
# 读取一帧关节和 IMU 数据并推理一次，无需手柄。
python e1_wheel/scripts/deploy_real_e1_wheel.py --read-only

# 用手柄控制真机。
python e1_wheel/scripts/deploy_real_e1_wheel.py
```

| 手柄 | 操作 |
| --- | --- |
| X | 从当前位置插值到默认站姿 |
| A | 到达 `STANDBY` 后开始策略推理 |
| Y | 进入阻尼 |
| B | 退出 |
| 左摇杆 | 前后和横移速度 |
| 右摇杆左右 | 转向速度 |

## 配置与策略

- [wheel_config.py](e1_wheel/wheel_config.py) 保存两边共用的默认模型路径、关节顺序、观测缩放、动作换算、默认站姿和 Kp/Kd。
- [robot_config.json](e1_wheel/scripts/robot_config.json) 保存真机地址、端口、手柄参数、关节方向、零偏和 IMU 对齐。关节配置每行依次是服务名、SDK 电机名、方向和零偏（rad）。
- `e1_wheel/model/` 保存当前的 `policy.onnx` 和 `policy.onnx.data`。换策略时，从同一次训练导出目录成对复制两个文件并保留原名。ONNX Runtime 会根据模型内部的引用加载同目录的外部权重，替换后需要重启运行进程。
- 仿真可用 `--policy /path/to/policy.onnx` 临时指定模型。实机可用 `--config` 指定另一份 JSON，该文件可以包含可选的 `policy` 字段。相对模型路径以该 JSON 所在目录为基准。
- `e1_wheel/robot/` 保存 URDF、网格和 MuJoCo 场景。`e1_wheel/build_mujoco_scene.py` 用于从训练 URDF 重建场景。

## 离线测试

```bash
python -m unittest discover -s e1_wheel/tests -v
```
