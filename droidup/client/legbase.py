# Copyright (c) 2025 DroidUp, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the DroidUp Software
# Development Kit License (20250922-DROIDUPSDK-SL).

import grpc
from typing import List, Optional
from droidup.api import droidup_msg_pb2
from droidup.api import leg_service_pb2_grpc
from droidup.client.grpcbase import GrpcBase, NanoSleep


class LegBase(GrpcBase):
    """机器人腿部控制SDK客户端
    提供简单易用的API接口来控制机器人腿部，封装底层gRPC调用。
    """
    def __init__(self, _host: str = "192.168.254.100", _port: int = 50052):
        super().__init__(_host= _host, _port=_port, _timeout=5.0, _retry_count=3, _retry_interval=1.0)
        print("Initializing Leg Client")
        # 初始化默认命令请求对象
        self._command_request = droidup_msg_pb2.DroidCommandRequest()
        self.stub = leg_service_pb2_grpc.LegServiceStub(self.channel)
        self._connected = True
        # 获取并显示机器人配置信息
        try:
            self.get_joint_config()
            self.get_joint_state()
            print("Successfully connected to: ", self.address)
        except Exception as e:
            print(f"Connection failed : {e}")
        self.show_config()
        # 初始化命令请求对象的所有字段
        self.init_command_request()
    
    def get_joint_config(self):
        """获取机器人腿部配置信息
        
        Returns:
            Dict: 包含关节名称、限位等配置信息
        
        Raises:
            RuntimeError: 如果未连接或请求失败
        """
        if not self._connected:
            raise RuntimeError("未连接到机器人控制服务器")
        
        # 如果已经缓存了配置信息，直接返回
        if self._configs:
            return self._configs
        
        try:
            self._configs = self.stub.GetLegConfig(droidup_msg_pb2.Empty())
            # 获取机器人配置中的关节数量，如果未连接则使用默认值
            self.num_actions = len(self._configs.joint_name)
            return self._configs
        except grpc.RpcError as e:
            self._last_error = f"gRPC错误: {e.code()}: {e.details()}"
            raise RuntimeError(f"获取配置失败: {self._last_error}")
        except Exception as e:
            raise RuntimeError(f"获取机器人配置失败: {e}")
    
    def get_joint_state(self):
        """获取机器人腿部当前状态
        
        Returns:
            Dict: 包含位置、速度、力矩等状态信息
        
        Raises:
            RuntimeError: 如果未连接或请求失败
        """
        if not self._connected:
            raise RuntimeError("未连接到机器人控制服务器")
        
        try:
            self._state = self.stub.GetLegState(droidup_msg_pb2.Empty())
            return self._state
        except Exception as e:
            raise RuntimeError(f"获取机器人状态失败: {e}")

    def set_joint_positions(self, positions: List[float], velocitys: Optional[List[float]] = None,
                            torques: Optional[List[float]] = None) -> bool:
        """设置关节位置控制命令

        Args:
            positions: 目标关节位置列表（rad）
            velocitys: 可选目标速度列表(rad/s)
            torques:   可选前馈力矩列表(N.m)

        Returns:
            bool: 命令是否成功发送

        Raises:
            RuntimeError: 如果未连接或请求失败
        """
        if not self._connected:
            raise RuntimeError("未连接到机器人控制服务器")
        try:
            for i in range(self.num_actions):
                self._command_request.position[i] = positions[i]

            for i in range(self.num_actions):
                self._command_request.velocity[i] = 0.0
                self._command_request.torque[i] = 0.0

            if velocitys:
                for i in range(self.num_actions):
                    self._command_request.velocity[i] = velocitys[i]

            if torques:
                for i in range(self.num_actions):
                    self._command_request.torque[i] = torques[i]

            # 发送命令
            # print(f"发送命令: {self._command_request}")
            self.stub.SetLegCommand(self._command_request)
            return True
        except Exception as e:
            print(f"设置关节位置失败: {e}")
            return False

