# p1_final_assembly.py
# 总装工厂：将各零件组装成 (B, target_channels, H, W) 张量
# 改进版：支持安全修剪（自动截取多余通道）和补零

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class P1FinalAssembly:
    """
    总装工厂：将多个零件焊接为最终多通道张量。
    输出格式：(B, target_channels, H, W)
    核心：每个零件应提供 `tensor` 字段，形状 (B, C, H, W)，C 应与分配区间长度匹配（若不匹配，可自动修剪或补零）。
    """

    def __init__(self,
                 global_resolution: Tuple[int, int] = (512, 512),
                 device: str = 'cpu',
                 channel_allocation: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        参数:
            global_resolution: 全局输出分辨率 (H, W)
            device: 计算设备
            channel_allocation: 零件工厂名到通道区间 (start, end) 的映射（半开区间）。
                若未提供，使用默认 48 通道分配。
        """
        self.H, self.W = global_resolution
        self.device = torch.device(device)
        # 默认 48 通道分配（与你的设计完全一致）
        self.default_allocation = {
            "skin_factory":   (0, 16),   # 皮肤 16 通道
            "joint_factory":  (16, 32),  # 关节 16 通道
            "fabric_factory": (32, 40),  # 织物 8 通道
            "fluid_factory":  (40, 44),  # 流体 4 通道
            "tear_factory":   (44, 48),  # 撕裂 4 通道
        }
        self.allocation = channel_allocation or self.default_allocation

    def _align_tensor(self, tensor: torch.Tensor,
                      target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """将任意张量对齐到 (B, C, H, W)，不改变通道数"""
        if tensor is None:
            return None
        Bt, Ct, Ht, Wt = target_shape
        # 处理维度
        if tensor.ndim == 3:
            if tensor.shape[-1] in (1, 2, 3, 4, 5, 8, 16):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[-1] in (1, 2, 3, 4, 5, 8, 16) and tensor.shape[1] not in (1, 2, 3, 4, 5, 8, 16):
                tensor = tensor.permute(0, 3, 1, 2)
        # 空间尺寸插值
        if tensor.shape[2] != Ht or tensor.shape[3] != Wt:
            tensor = F.interpolate(tensor, size=(Ht, Wt), mode='bilinear', align_corners=False)
        # 批次对齐
        if tensor.shape[0] != Bt:
            if tensor.shape[0] > Bt:
                tensor = tensor[:Bt]
            else:
                tensor = tensor.repeat(Bt, 1, 1, 1)
        return tensor

    def weld(self,
             components: List[Dict[str, Any]],
             target_channels: int,
             target_batch: int = 1,
             clear_cache: bool = False,
             strict_channel_match: bool = True,
             normal_blend_mode: str = 'add') -> Dict[str, torch.Tensor]:
        """
        动态焊接所有零件，生成多通道张量。

        参数:
            components: 质检后的零件列表（每个元素是字典）
            target_channels: 最终输出的总通道数（应为 48）
            target_batch: 批次大小
            clear_cache: 是否清理 GPU 缓存
            strict_channel_match: 
                True  -> 通道数不匹配时报错
                False -> 通道数过多时自动截取，过少时补零（并打印警告）
            normal_blend_mode: 保留参数，本版本中未使用（区间不重叠，无需混合）
        返回:
            {"final_tensor": torch.Tensor}
        """
        B = target_batch
        H, W = self.H, self.W
        final_tensor = torch.zeros(B, target_channels, H, W, device=self.device)

        for comp in components:
            if comp is None:
                continue
            # 获取工厂名称
            factory_name = comp.get("factory", "")
            if factory_name not in self.allocation:
                # 向后兼容：尝试根据旧字段名推断工厂类型
                if "base_normal" in comp:
                    factory_name = "skin_factory"
                elif "joint_delta" in comp:
                    factory_name = "joint_factory"
                elif "fabric_tensor" in comp:
                    factory_name = "fabric_factory"
                elif "fluid_mask" in comp:
                    factory_name = "fluid_factory"
                elif "hole_mask" in comp:
                    factory_name = "tear_factory"
                else:
                    print(f"[P1FinalAssembly] 跳过未知零件，factory={factory_name}")
                    continue

            start, end = self.allocation.get(factory_name, (None, None))
            if start is None or end is None:
                print(f"[P1FinalAssembly] 工厂 {factory_name} 没有分配通道区间，跳过")
                continue
            if end > target_channels:
                print(f"[P1FinalAssembly] 零件 {factory_name} 区间 [{start}:{end}] 超出目标通道数 {target_channels}，跳过")
                continue

            # 优先使用零件的 `tensor` 字段（新接口）
            tensor = comp.get("tensor")
            if tensor is None:
                # 向后兼容：尝试根据工厂名获取旧字段
                if factory_name == "skin_factory":
                    tensor = comp.get("base_normal")
                elif factory_name == "joint_factory":
                    tensor = comp.get("normal_delta") or comp.get("joint_delta")
                elif factory_name == "fabric_factory":
                    tensor = comp.get("detail_normal") or comp.get("fabric_tensor")
                elif factory_name == "fluid_factory":
                    tensor = comp.get("fluid_normal_delta") or comp.get("fluid_mask")
                elif factory_name == "tear_factory":
                    tensor = comp.get("edge_normal_delta") or comp.get("hole_mask")
                else:
                    tensor = None

            if tensor is None:
                print(f"[P1FinalAssembly] 零件 {factory_name} 没有可用的张量，跳过")
                continue

            # 对齐张量形状
            target_shape = (B, tensor.shape[1], H, W)
            tensor = self._align_tensor(tensor, target_shape)
            if tensor is None:
                continue

            src_ch = tensor.shape[1]
            allocated_ch = end - start

            # 通道数处理：安全修剪或补零
            if src_ch != allocated_ch:
                msg = f"零件 {factory_name} 通道数 {src_ch} 与分配区间 [{start}:{end}] 长度 {allocated_ch} 不匹配"
                if strict_channel_match:
                    raise ValueError(msg)
                else:
                    if src_ch > allocated_ch:
                        # 只取前 allocated_ch 个通道
                        print(f"[P1FinalAssembly] 警告: {msg}，将截取前 {allocated_ch} 通道")
                        tensor = tensor[:, :allocated_ch, :, :]
                    else:
                        # 通道不足，补零
                        print(f"[P1FinalAssembly] 警告: {msg}，将在通道末尾补零")
                        pad = torch.zeros(B, allocated_ch - src_ch, H, W, device=self.device, dtype=tensor.dtype)
                        tensor = torch.cat([tensor, pad], dim=1)

            # 切片注入
            final_tensor[:, start:end, :, :] = tensor

        # 最终通道数对齐（确保输出 target_channels 通道）
        if final_tensor.shape[1] != target_channels:
            if final_tensor.shape[1] > target_channels:
                final_tensor = final_tensor[:, :target_channels, :, :]
            else:
                pad = torch.zeros(B, target_channels - final_tensor.shape[1], H, W,
                                  device=self.device, dtype=final_tensor.dtype)
                final_tensor = torch.cat([final_tensor, pad], dim=1)

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"final_tensor": final_tensor}