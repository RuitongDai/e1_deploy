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

默认加载 `e1_wheel/robot/E1_DOG_v2/scene.xml`（平地）。使用包含台阶和斜坡的场景：

```bash
python e1_wheel/scripts/sim2sim_e1_wheel.py --model e1_wheel/robot/E1_DOG_v2/scene_multi_terrain.xml
```

多地形场景通过 `include` 复用 `scene.xml`。出生点前方沿 +X 方向依次是上下台阶（x=2～5.6 m，每级高 8 cm、踏面深 40 cm，顶部平台高 32 cm）和上下斜坡（x=7～12 m，坡度约 11.31°、平台高 40 cm），宽度均为 2 m；两段之间保留平地。此命令不要加 `--rebuild-model`，该选项仅支持默认场景。

**真机**

先核对 [robot_config.json](e1_wheel/scripts/robot_config.json) 里的 SDK 地址、关节映射和控制模式。当前默认是前腿及前轮 `192.168.51.3:50054`，后腿及后轮 `192.168.51.3:50053`。

```bash
# 读取一帧关节和 IMU 数据并推理一次，无需手柄。
python e1_wheel/scripts/deploy_real_e1_wheel.py --read-only

# 用手柄控制真机。
python e1_wheel/scripts/deploy_real_e1_wheel.py
```

排查零速度指令下的前移时，可先让机器人机身保持静止，连续读取后腿控制器 IMU：

```bash
python e1_wheel/scripts/deploy_real_e1_wheel.py --monitor-imu --duration 10
```

此模式仅调用 `GetLegState`，不加载策略、不使用手柄、不发送电机命令，也不会停止其他进程的控制。输出原始欧拉角、按部署弧度约定换算的角度、机身系角速度和投影重力、原始加速度、IMU 时间戳，以及静止采样均值/标准差。时间戳单位和原始加速度单位由固件决定。机身水平静止且 IMU 坐标对齐时，策略输入的 `gyro_body` 应接近 `[0,0,0]`，`gravity_body` 应接近 `[0,0,-1]`；实际机身倾角会反映在重力分量中。部署代码按欧拉角 rad、角速度 rad/s 解释返回值，需要核对固件单位。

正常控制时增加 `--debug-state` 可每秒打印两次状态、手柄输入、策略速度指令 `cmd`、IMU 输入及四轮目标/实际速度（关节顺序 FR、FL、RR、RL，轮速单位 rad/s）。该选项仍是正常电机控制模式。若 `cmd` 为零但 `wheel_target` 持续非零，说明策略仍输出轮速，应结合 IMU 对齐、关节零偏、实际姿态和仿真表现排查。

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
- 仿真和真机的速度范围统一由 `wheel_config.py` 中的 `COMMAND_LIMIT = np.array([2.0, 1.0, 1.5])` 设置，顺序为前后速度（m/s）、横移速度（m/s）、转向角速度（rad/s）。摇杆归一化输入乘以此数组，所以推满分别得到 ±2 m/s、±1 m/s、±1.5 rad/s，推一半得到对应的一半；修改后需重启脚本。`robot_config.json` 的旧字段 `command_scale` 已移除，自定义 JSON 中若仍保留该字段也不再使用。`COMMAND_RATE_LIMIT_ENABLED` 控制加减速平滑，不改变最终速度范围。
- [robot_config.json](e1_wheel/scripts/robot_config.json) 保存真机地址、端口、手柄参数、关节方向、零偏和 IMU 对齐。关节配置每行依次是服务名、SDK 电机名、方向和零偏（rad）。
- `e1_wheel/model_rough/` 保存当前的 `policy.onnx` 和 `policy.onnx.data`。换策略时，从同一次训练导出目录成对复制两个文件并保留原名。ONNX Runtime 会根据模型内部的引用加载同目录的外部权重，替换后需要重启运行进程。
- 仿真可用 `--policy /path/to/policy.onnx` 临时指定模型。实机可用 `--config` 指定另一份 JSON，该文件可以包含可选的 `policy` 字段。相对模型路径以该 JSON 所在目录为基准。
- `e1_wheel/robot/` 保存 URDF、网格和 MuJoCo 场景。`e1_wheel/build_mujoco_scene.py` 用于从训练 URDF 重建场景。

## 离线测试

```bash
python -m unittest discover -s e1_wheel/tests -v
```
