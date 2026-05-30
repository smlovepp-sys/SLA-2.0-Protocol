# p1_external_normal.py
# 外部法线内核加载器与通用调用接口

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class ExternalNormalKernel:
    """
    管理外部法线内核的加载、存储和推理。
    若 normal_calc_kernel.pth 存在则加载，否则为空。
    """
    def __init__(self, kernel_path: Optional[str] = None):
        self.kernel = None
        self.config = {}
        self._load_kernel(kernel_path)

    def _load_kernel(self, kernel_path: Optional[str]):
        if kernel_path is None:
            kernel_path = os.path.join(os.path.dirname(__file__), "normal_calc_kernel.pth")
        if not os.path.exists(kernel_path):
            return
        try:
            checkpoint = torch.load(kernel_path, map_location='cpu')
            state_dict = checkpoint.get("model_state", checkpoint)
            self.config = checkpoint.get("config", {})
            # 构建一个简单的自适应网络
            in_ch = self.config.get("input_channels", 2)
            out_ch = self.config.get("output_channels", 3)
            self.kernel = SimpleNormalNet(in_ch, out_ch)
            self.kernel.load_state_dict(state_dict, strict=False)
            self.kernel.eval()
            for param in self.kernel.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"[ExternalNormalKernel] 加载失败: {e}，将使用默认过程式生成")
            self.kernel = None

    def to(self, device: torch.device):
        if self.kernel is not None:
            self.kernel.to(device)

    def is_available(self) -> bool:
        return self.kernel is not None

    def apply(self, input_tensor: torch.Tensor, target_channels: int = 3) -> Optional[torch.Tensor]:
        """
        将内核应用于输入张量，返回增强后的法线图。
        若内核不可用或推理失败，返回 None。
        """
        if self.kernel is None:
            return None
        try:
            device = input_tensor.device
            self.kernel.to(device)
            with torch.no_grad():
                # 如果输入通道与内核期望不符，临时投影
                in_ch = input_tensor.shape[1]
                expected_in = self.config.get("input_channels", 2)
                if in_ch != expected_in:
                    proj = nn.Conv2d(in_ch, expected_in, 1, bias=False).to(device)
                    nn.init.constant_(proj.weight, 1.0 / in_ch)
                    x = proj(input_tensor)
                else:
                    x = input_tensor
                out = self.kernel(x)
                if out.shape[1] != target_channels:
                    proj_out = nn.Conv2d(out.shape[1], target_channels, 1, bias=False).to(device)
                    nn.init.constant_(proj_out.weight, 1.0)
                    out = proj_out(out)
                # 确保输出值域在 [0,1] 或 [-1,1] 由调用方决定，这里不做强制钳位
                return out
        except Exception as e:
            print(f"[ExternalNormalKernel] 推理失败: {e}")
            return None


class SimpleNormalNet(nn.Module):
    """一个轻量级法线生成网络，用于外部内核参考"""
    def __init__(self, in_ch=2, out_ch=3, base_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x