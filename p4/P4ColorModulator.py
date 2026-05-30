# P4ColorModulator.py (全通路激活 + 动态运动融合 + 负数防护 + 时序残差 + 物理去色 + LeakyReLU + 张量完整性检查)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class P4ColorModulator(nn.Module):
    def __init__(self,
                 in_channels_physics: int = 48,  # geo(16) + mat(16) + det(16) = 48
                 hidden: int = 32,
                 out_channels: int = 3,
                 use_rule_based_prior: bool = True):
        super().__init__()
        self.use_rule_based_prior = use_rule_based_prior

        self.fixed_in_channels = 3 + in_channels_physics + 3
        
        self.conv1 = nn.Conv2d(self.fixed_in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        
        self.conv4 = nn.Conv2d(hidden, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)
        # LeakyReLU 替代 ReLU
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    @staticmethod
    def check_tensor_integrity(tensor: torch.Tensor, name: str, threshold: float = 10.0):
        """检查张量中是否存在 NaN/Inf 或极端值，输出诊断信息"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"[CRITICAL] {name} 出现 NaN/Inf!")
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        if abs(max_val) > threshold or abs(min_val) > threshold:
            print(f"[WARN] {name} 数值异常: Max={max_val:.4f}, Min={min_val:.4f} | 可能发生梯度爆炸")
        if tensor.dim() == 4:
            channel_means = tensor.abs().mean(dim=(0, 2, 3))
            if channel_means.max() > threshold / 2:
                idx = torch.argmax(channel_means)
                print(f"[DEBUG] {name} 中第 {idx} 通道均值过高，可能是过曝源头")

    def rule_based_correction(self,
                              rgb: torch.Tensor,
                              geo: torch.Tensor,
                              mat: torch.Tensor) -> torch.Tensor:
        wetness_map = mat[:, 0:1]
        stress_map = geo[:, 0:1]
        sat_gain = 1.0 + wetness_map * 0.4
        sat_gain = torch.clamp(sat_gain, 0.9, 1.3)
        cont_gain = 1.0 + stress_map * 0.3
        cont_gain = torch.clamp(cont_gain, 0.95, 1.2)
        gray = rgb.mean(dim=1, keepdim=True)
        rgb_adj = gray + (rgb - gray) * sat_gain
        rgb_adj = 0.5 + (rgb_adj - 0.5) * cont_gain
        return rgb_adj.clamp(0, 1)

    def forward(self,
                curr_rgb: torch.Tensor,
                geo: torch.Tensor,
                mat: torch.Tensor,
                det: torch.Tensor,
                prev_rgb: torch.Tensor,
                metadata: Optional[Dict[str, float]] = None) -> torch.Tensor:
        
        # 入口数值检查
        self.check_tensor_integrity(curr_rgb, "curr_rgb")
        self.check_tensor_integrity(prev_rgb, "prev_rgb")
        self.check_tensor_integrity(geo, "geo")
        self.check_tensor_integrity(mat, "mat")
        self.check_tensor_integrity(det, "det")

        # --- 物理场“去色”压制：材质通道只保留微弱光泽信息，不影响颜色 ---
        mat = torch.tanh(mat * 0.2) * 0.1
        
        # 🛡️ 入口防护：斩断负数毒素
        geo = torch.clamp(geo, min=0.0)
        det = torch.clamp(det, min=0.0)
        curr_rgb = torch.clamp(curr_rgb, min=0.0, max=1.0)
        prev_rgb = torch.clamp(prev_rgb, min=0.0, max=1.0)

        H_target, W_target = curr_rgb.shape[-2:]
        if geo.shape[-2:] != (H_target, W_target):
            geo = F.interpolate(geo, size=(H_target, W_target), mode='bilinear', align_corners=False)
        if mat.shape[-2:] != (H_target, W_target):
            mat = F.interpolate(mat, size=(H_target, W_target), mode='bilinear', align_corners=False)
        if det.shape[-2:] != (H_target, W_target):
            det = F.interpolate(det, size=(H_target, W_target), mode='bilinear', align_corners=False)
        if prev_rgb.shape[-2:] != (H_target, W_target):
            prev_rgb = F.interpolate(prev_rgb, size=(H_target, W_target), mode='bilinear', align_corners=False)

        # 运动感知
        pixel_motion = torch.mean(torch.abs(curr_rgb - prev_rgb), dim=1, keepdim=True)
        stress_map = geo[:, 0:1]
        combined_motion = torch.clamp(pixel_motion * 4.0 + stress_map * 0.5, 0.0, 1.0)

        # 规则先验调制
        if self.use_rule_based_prior:
            base_rgb = self.rule_based_correction(curr_rgb, geo, mat)
        else:
            base_rgb = curr_rgb

        self.check_tensor_integrity(base_rgb, "base_rgb")

        # 卷积残差调制
        x = torch.cat([base_rgb, geo, mat, det, prev_rgb], dim=1)
        if x.shape[1] < self.fixed_in_channels:
            pad = torch.zeros(x.shape[0], self.fixed_in_channels - x.shape[1],
                              H_target, W_target, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.fixed_in_channels:
            x = x[:, :self.fixed_in_channels, :, :]

        self.check_tensor_integrity(x, "concat_x")

        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        feat = self.relu(self.conv3(feat))
        residual = self.conv4(feat)

        self.check_tensor_integrity(residual, "residual")
        modulated_rgb = (base_rgb + residual).clamp(0, 1)
        self.check_tensor_integrity(modulated_rgb, "modulated_rgb")

        # 时序残差融合：避免直接连乘/平均导致的亮度崩塌
        final_rgb = modulated_rgb + 0.05 * (prev_rgb - modulated_rgb)
        final_rgb = final_rgb.clamp(0, 1)
        self.check_tensor_integrity(final_rgb, "final_rgb")

        return final_rgb