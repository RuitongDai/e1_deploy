# Copyright (c) 2025 DroidUp, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the DroidUp Software
# Development Kit License (20250922-DROIDUPSDK-SL).

import grpc
import math
import time
import logging
from typing import List, Optional
# ------------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class GrpcBase:
    """卓益得机器人GRPC通信基类, DroidUp GRPC communication base class
    """

    def __init__(self, _host: str = None, _port: int = None, _timeout: float = None,
                       _retry_count: int = None, _retry_interval: float = None):
        """初始化机器人手臂客户端

        Args:
            _host: 自定义主机地址
            _port: 自定义端口
            _timeout: 连接超时时间（秒）
            _retry_count: 连接重试次数
            _retry_interval: 重试间隔（秒）
        """
        self.address = f"{_host}:{_port}"

        self._configs = None
        self.num_actions = 5  # 默认关节数量
        self._command_request = None

        # 连接参数
        self.timeout = _timeout
        self.retry_count = _retry_count
        self.retry_interval = _retry_interval

        # 初始化其他属性
        self.channel = None
        self.stub = None
        self._state = None
        self._connected = False
        self._state_thread = None
        self._running = False
        self._last_error = None
        self._logger = logging.getLogger(__name__)
        self.channel = grpc.insecure_channel(self.address)

    def connect(self) -> bool:
        """连接到机器人控制服务器，支持重试机制

        Returns:
            bool: 连接是否成功
        """

        if self._connected:
            self._logger.warning("Already connected to Arm.")
            return True


        self._connected = True
        return True

    def disconnect(self):
        """断开与机器人控制服务器的连接"""
        if self._state_thread and self._running:
            self._running = False
            self._state_thread.join(timeout=1.0)

        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
            self._connected = False

    def is_connected(self) -> bool:
        """检查是否已连接到机器人控制服务器

        Returns:
            bool: 是否已连接
        """
        return self._connected

    def get_joint_config(self):
        pass

    def get_joint_state(self):
        pass

    def init_command_request(self):
        """初始化命令请求对象的所有字段

        为DroidCommandRequest的所有字段设置合适的默认值，确保发送到C++服务端的请求对象完全初始化。
        这个方法应该在每次发送新命令前调用，替代Clear()方法。
        """
        # 清空之前的数据
        self._command_request.Clear()

        # 为每个关节初始化所有字段
        for _ in range(self.num_actions):
            self._command_request.mode.append(1)  # 默认位置控制模式
            self._command_request.position.append(0.0)
            self._command_request.velocity.append(0.0)
            self._command_request.torque.append(0.0)
            self._command_request.ens.append(1)  # 默认启用
            self._command_request.kp.append(0.1)  # 默认位置增益
            self._command_request.kd.append(0.1)  # 默认阻尼增益
            self._command_request.max_torque.append(1.0)  # 默认最大力矩限制

    def set_joint_mode(self, modes: List[float]):
        for i in range(self.num_actions):
            self._command_request.mode[i] = modes[i]

    def set_joint_enable_state(self, ens: List[float]):
        for i in range(self.num_actions):
            self._command_request.ens[i] = ens[i]

    def set_joint_parameters(self, kp: Optional[List[float]] = None, kd: Optional[List[float]] = None,
                             max_torque: Optional[List[float]] = None):
        if kp:
            for i in range(self.num_actions):
                self._command_request.kp[i] = kp[i]
        if kd:
            for i in range(self.num_actions):
                self._command_request.kd[i] = kd[i]
        if max_torque:
            for i in range(self.num_actions):
                self._command_request.max_torque[i] = max_torque[i]

    def set_joint_path(self, _time, _qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        qt = [0.0] * self.num_actions
        state = self.get_joint_state()
        q0 = state.position
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < _time + dt / 2.0:
            start_time = time.perf_counter()
            st = min(tt / _time, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0

            for idx in range(self.num_actions):  # 假设关节数量是18
                qt[idx] = s0 * q0[idx] + s1 * _qd[idx]
            self.set_joint_positions(qt)
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def emergency_stop(self) -> bool:
        """紧急停止所有关节运动

        Returns:
            bool: 命令是否成功发送
        """
        if not self._connected:
            raise RuntimeError("未连接到机器人控制服务器")

        try:
            self.set_joint_enable_state([False] * self.num_actions)
            # 发送命令
            print(f"发送命令: {self._command_request}")
            return True
        except Exception as e:
            print(f"紧急停止失败: {e}")
            return False

    def show_config(self):
        print("number off dof:", self.num_actions)
        line = '-' * self.num_actions * 11
        print("---------+" + line)
        print("MtrName  |", end="")
        for idx in range(self.num_actions):
            print(f"{self._configs.joint_name[idx]:>11}", end="")
        print()
        print("---------+" + line)
        for attr in ["pzero", "pmin", "pmax", "imax", "kp", "kd"]:
            print("{:<8} |".format(attr.upper()), end="")
            for i in range(self.num_actions):
                print("{:>11.3f}".format(getattr(self._configs, attr)[i]), end="")
            print()

        print("---------+" + line)

class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break