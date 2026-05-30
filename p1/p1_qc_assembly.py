# p1_qc_assembly.py
# P1 质检厂与装配厂（纯张量逻辑版）
# 修复：维度转换逻辑更加严密，避免重复翻转

import torch
from typing import Dict, List, Any, Optional, Tuple, Union


class P1QCInspector:
    """
    P1 质检厂：纯张量逻辑哨兵，不进行任何 IO 或可视化。
    核心功能：
        1. 零件身份核对（factory, component_id）
        2. 全维度对齐检查（B, C, H, W）
        3. NaN/Inf 扫描与修复（替换为 0 并记录）
        4. 数值范围钳位（法线 [0,1]）
        5. 设备统一（所有张量移至 target_device）
    """

    def __init__(self, target_device: Union[str, torch.device] = 'cpu'):
        self.target_device = torch.device(target_device)
        self.warnings = []

    def _check_component_id(self, component: Dict[str, Any]) -> bool:
        if 'component_id' not in component:
            self.warnings.append("缺少 component_id，零件被拒收")
            return False
        if 'factory' not in component:
            self.warnings.append(f"零件 {component['component_id']} 缺少 factory 标签，拒收")
            return False
        return True

    def _fix_tensor_shape(self, tensor: torch.Tensor, expected_ndim: int = 4) -> torch.Tensor:
        """
        尝试将张量修正为 (B, C, H, W) 格式。
        改进：更准确地判断形状语义，避免将已经是 (B,C,H,W) 的张量误转换为 (B,H,W,C) 再转回来。
        """
        if tensor is None:
            return None
        ndim = tensor.ndim
        if ndim == expected_ndim:
            # 已经是4维，但可能是 (B, H, W, C) 格式，需要判断通道维度位置
            # 如果最后一个维度很小（如 ≤5）且第二个维度很大（≥100），则可能是 (B,H,W,C)
            shape = tensor.shape
            if shape[-1] in (1, 2, 3, 4, 5, 48) and shape[1] not in (1, 2, 3, 4, 5, 48):
                # 假设是 (B, H, W, C) -> 转置为 (B, C, H, W)
                self.warnings.append(f"检测到 (B,H,W,C) 格式 ({shape})，转换为 (B,C,H,W)")
                tensor = tensor.permute(0, 3, 1, 2)
            # 否则保持原样
            return tensor

        if ndim == 3:
            # 3维张量可能是 (C, H, W) 或 (H, W, C)
            # 判断依据：如果最后一个维度很小（≤5），则很可能是 (H,W,C)
            shape = tensor.shape
            if shape[-1] in (1, 2, 3, 4, 5, 48):
                # (H, W, C) -> (C, H, W)
                tensor = tensor.permute(2, 0, 1)
                self.warnings.append(f"3D张量从 (H,W,C) 转换为 (C,H,W)")
            else:
                # 假设已经是 (C, H, W)
                pass
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            self.warnings.append(f"张量形状从 {ndim}D 修正为 4D")
            return tensor

        if ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            self.warnings.append(f"张量形状从 2D 修正为 4D")
            return tensor

        self.warnings.append(f"无法修正的形状 {tensor.shape}，跳过")
        return None

    def _validate_normal_channels(self, tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
        """法线通道处理：1通道复制为3，多通道直接通过"""
        if tensor is None:
            return None
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
            self.warnings.append(f"{tensor_name} 通道数为1，已复制为3通道")
        elif tensor.shape[1] > 3:
            self.warnings.append(f"{tensor_name} 通道数为 {tensor.shape[1]}，已接受（未截断）")
        return tensor

    def _check_nan_inf(self, tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
        if tensor is None:
            return None
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.warnings.append(f"{tensor_name} 包含 NaN/Inf，已替换为 0")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def _clamp_normal(self, tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
        if tensor is None:
            return None
        if tensor.min() < 0 or tensor.max() > 1:
            self.warnings.append(f"{tensor_name} 数值超出 [0,1]，已钳位")
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor

    def _align_device(self, tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
        if tensor is None:
            return None
        if tensor.device != self.target_device:
            self.warnings.append(f"{tensor_name} 设备从 {tensor.device} 移至 {self.target_device}")
            tensor = tensor.to(self.target_device)
        return tensor

    def inspect_component(self, component: Dict[str, Any],
                          expected_batch: Optional[int] = None,
                          expected_hw: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        self.warnings = []
        if not self._check_component_id(component):
            return None

        cleaned = component.copy()
        tensor_fields = ['base_normal', 'initial_normal', 'joint_delta', 'rig_map',
                         'normal_delta', 'fabric_tensor', 'alpha_mask', 'detail_normal',
                         'fluid_mask', 'fluid_normal_delta', 'refraction_data',
                         'hole_mask', 'edge_debris', 'edge_normal_delta']

        for field in tensor_fields:
            if field in cleaned:
                tensor = cleaned[field]
                if tensor is None:
                    continue
                tensor = self._fix_tensor_shape(tensor, expected_ndim=4)
                if tensor is None:
                    cleaned[field] = None
                    continue
                if 'normal' in field.lower():
                    tensor = self._validate_normal_channels(tensor, field)
                if expected_batch is not None and tensor.shape[0] != expected_batch:
                    self.warnings.append(f"{field} batch={tensor.shape[0]} 期望 {expected_batch}，将截取/重复")
                    if tensor.shape[0] > expected_batch:
                        tensor = tensor[:expected_batch]
                    else:
                        repeats = [expected_batch // tensor.shape[0] + 1, 1, 1, 1]
                        tensor = tensor.repeat(*repeats)[:expected_batch]
                if expected_hw is not None and (tensor.shape[2] != expected_hw[0] or tensor.shape[3] != expected_hw[1]):
                    self.warnings.append(f"{field} 尺寸 {tensor.shape[2:]} 期望 {expected_hw}，将插值")
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=expected_hw, mode='bilinear', align_corners=False
                    )
                tensor = self._check_nan_inf(tensor, field)
                if 'normal' in field.lower():
                    tensor = self._clamp_normal(tensor, field)
                tensor = self._align_device(tensor, field)
                cleaned[field] = tensor

        if self.warnings:
            cleaned['qc_warnings'] = self.warnings.copy()
        return cleaned


class P1Assembler:
    """
    P1 装配厂：接收质检后的零件包，按 depth_rank 排序，生成最终输出包。
    """
    def __init__(self):
        pass

    def assemble(self, validated_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid = [c for c in validated_components if c is not None]
        depth_map = {}
        components_dict = {}
        for comp in valid:
            cid = comp.get('component_id')
            if not cid:
                continue
            components_dict[cid] = comp
            depth = 0
            if 'alignment_meta' in comp and isinstance(comp['alignment_meta'], dict):
                depth = comp['alignment_meta'].get('layer_depth', 0)
            elif 'slot_6_physics_layer' in comp:
                depth = comp['slot_6_physics_layer'].get('depth_rank', 0)
            depth_map[cid] = depth
        render_order = sorted(depth_map.keys(), key=lambda x: depth_map[x])
        return {
            "p1_version": "v7_qc_assembly",
            "render_order": render_order,
            "components": components_dict
        }


if __name__ == "__main__":
    # 测试：模拟一个已经是 (B,C,H,W) 的张量
    good_tensor = torch.rand(1, 48, 512, 512)
    comp = {"component_id": "test", "factory": "test_factory", "base_normal": good_tensor}
    qc = P1QCInspector()
    cleaned = qc.inspect_component(comp, expected_batch=1, expected_hw=(512,512))
    print("警告:", cleaned.get('qc_warnings', []))
    print("最终形状:", cleaned['base_normal'].shape)