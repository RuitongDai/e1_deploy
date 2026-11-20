# E1 Dog Sim2Real 环境配置（uv）

轮足 E1 的 RobotLab → MuJoCo 仿真部署见 [e1_wheel/README.md](e1_wheel/README.md)。

本项目用于部署 E1 Dog 16 自由度 ONNX 策略。使用 **Python 3.10**，通过 **uv** 在项目目录创建名为 **`e1_deploy`** 的虚拟环境。

以下命令适用于 Linux / Bash。部署脚本使用 `termios`、`tty` 等 Unix 接口，不能直接在原生 Windows Python 中运行。当前推理代码明确使用 ONNX Runtime 的 CPU 后端，无需安装 CUDA、PyTorch 或训练环境。

## 1. 安装 uv

已安装 uv 可以跳过安装命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

参考：[uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。

## 2. 创建并激活 e1_deploy

进入项目根目录（包含两个 `.whl` 文件和部署脚本的目录）：

```bash
cd /home/dai/droidup/e1_dog_sim2real

# 在当前目录创建 e1_deploy/，终端提示符也显示 e1_deploy。
uv venv --python 3.10 --prompt e1_deploy e1_deploy
source e1_deploy/bin/activate

python --version
python -c "import sys; print(sys.executable)"
```

如果项目位于其他位置，修改 `cd` 路径。没有可用的 Python 3.10 时，uv 默认会自动下载。已有 `e1_deploy/` 时直接激活即可，不必重复创建。

确认 Python 版本为 `3.10.x`，解释器路径以 `/e1_deploy/bin/python` 结尾。

参考：[uv 虚拟环境说明](https://docs.astral.sh/uv/pip/environments/)。

## 3. 安装依赖

在项目根目录执行：

```bash
uv pip install --python e1_deploy/bin/python -r requirements.txt
uv pip check --python e1_deploy/bin/python
```

安装命令显式指定了解释器，即使未激活环境，也会安装到 `e1_deploy`。

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| numpy | 1.26.4 | 观测与动作数组计算 |
| onnx | 1.16.2 | 读取、检查策略模型 |
| onnxruntime | 1.19.2 | CPU 策略推理 |
| pygame | 2.6.1 | 手柄输入与手柄检测工具 |
| droidup-api | 1.0.0 | 机器人通信协议，本地 wheel |
| droidup-client | 1.0.0 | 机器人客户端，本地 wheel |

两个 SDK 文件必须保留在项目根目录：

```text
droidup_api-1.0.0-py3-none-any.whl
droidup_client-1.0.0-py3-none-any.whl
```

SDK 的包元数据要求 `protobuf==5.26.1`、`grpcio==1.65.4`、`grpcio-tools==1.65.4`，uv 会自动安装。`requirements.txt` 固定了直接依赖版本；传递依赖仍由 uv 解析。

## 4. 验证环境与策略

先检查依赖导入；这里只导入 SDK 类，不创建机器人连接：

```bash
source e1_deploy/bin/activate
python - <<'PY'
import sys
import numpy
import onnx
import onnxruntime
import pygame
from droidup.client.armbase import ArmBase
from droidup.client.legbase import LegBase
from droidup.client.grpcbase import NanoSleep

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("ONNX:", onnx.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
print("依赖导入成功，CPU 推理后端可用")
PY
```

再验证目录内的 `policy_1.onnx`：

```bash
python sim2real_e1_dog_16dof.py \
  --policy ./policy_1.onnx \
  --validate-policy-only
```

该命令检查模型维度并执行一次推理，不连接机器人。预期输入为 `(1, 342)`、输出为 `(1, 16)`，日志包含 `backend=onnxruntime` 和 `ONNX policy validation passed`。如果日志显示 `onnx.reference (validation only)`，说明没有使用 ONNX Runtime，请重新检查其安装；真机运行要求 ONNX Runtime。

## 5. 运行入口

### 手柄测试

连接手柄后运行：

```bash
python gamepad_controller.py
```

该工具只检测手柄，不连接机器人。推动摇杆、按下按钮查看编号，按 `Ctrl+C` 退出。部署脚本的映射以 `gamepad_controller.py` 中的实现为准。

### 真机运行

运行前需要机器人 SDK 服务已启动，部署机可以访问机器人 IP 和两个端口。模型还应先在对应的 MuJoCo 仿真中验证；本目录不包含该仿真环境。

默认地址为 `192.168.254.100`，前腿服务端口为 `50054`，后腿服务端口为 `50053`。按实际网络配置修改参数。

手柄模式：

```bash
python sim2real_e1_dog_16dof.py \
  --policy ./policy_1.onnx \
  --host 192.168.254.100 \
  --upper-port 50054 \
  --lower-port 50053 \
  --gamepad
```

键盘模式（需要交互式终端；SSH 使用 `ssh -t`）：

```bash
python sim2real_e1_dog_16dof.py \
  --policy ./policy_1.onnx \
  --host 192.168.254.100 \
  --keyboard --no-gamepad
```

| 操作 | 手柄 | 键盘 |
| --- | --- | --- |
| 插值到默认站姿 | X | Z |
| 从待机进入策略推理 | A | X |
| 进入阻尼模式 | Y | C |
| 阻尼停止并退出 | B | Q |
| 前进 / 后退 | 左摇杆上下 | W / S |
| 左移 / 右移 | 左摇杆左右 | A / D |
| 左转 / 右转 | 转向摇杆 | J / L |
| 速度指令归零 | 摇杆回中 | R |

键盘速度指令按次累加，松开按键不会自动归零，按 `R` 清零。更多参数可通过 `python sim2real_e1_dog_16dof.py --help` 查看。

## 6. 日常使用与常见问题

每次新开终端后：

```bash
cd /home/dai/droidup/e1_dog_sim2real
source e1_deploy/bin/activate
```

退出虚拟环境：

```bash
deactivate
```

- **提示找不到虚拟环境或模块**：激活 `e1_deploy`，或直接使用 `./e1_deploy/bin/python` 运行脚本。`e1_deploy` 是自定义目录名，安装依赖时保留上文的 `--python e1_deploy/bin/python`。
- **uv 缓存目录只读 / `Could not acquire lock`**：先执行 `export UV_CACHE_DIR="${TMPDIR:-/tmp}/e1-deploy-uv-cache"`，再重试 uv 命令。
- **`pypi.org` 域名解析失败**：依赖尚未安装完成，确认部署机的 DNS 和网络可访问 PyPI 后，重新执行第 3 步。
- **依赖版本冲突**：使用独立的 Python 3.10 环境并按 `requirements.txt` 安装，保留 SDK 要求的 gRPC / Protobuf 版本。
- **未检测到手柄**：先运行手柄测试工具，确认系统能识别设备且当前用户有访问权限；也可以使用键盘模式。
- **机器人连接失败**：确认 `--host`、两个服务端口及 SDK 服务状态。策略验证通过只代表本地模型与推理可用，不代表真机连接已通过。
