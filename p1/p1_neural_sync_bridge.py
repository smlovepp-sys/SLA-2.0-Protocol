# p1_neural_sync_bridge.py
# P1 神经同步桥接器：空间对齐、设备锁死、零件捕获与同步包封装
# 修复：expected_factories 为 None 时不崩溃，占位符不计入 OK 状态

import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional


class P1NeuralSyncBridge:
    """
    桥接器：负责将所有工厂产出的零件同步到统一的物理空间。
    核心操作：
        1. 捕获零件，缺失零件创建空占位符
        2. 强制所有张量对齐到全局分辨率（插值缩放）
        3. 统一设备与数值精度
        4. 汇总有效区域掩模（mask_collective）
        5. 输出同步包，不修改数值内容（除必要的几何变换外）
    """

    def __init__(self, target_resolution: Tuple[int, int],
                 target_device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 fill_placeholder: bool = True):
        """
        参数：
            target_resolution: 全局目标分辨率 (H, W)
            target_device: 所有张量将移至的目标设备
            dtype: 统一的数据类型
            fill_placeholder: 是否为主模块要求但缺失的零件创建空占位符
        """
        self.target_resolution = target_resolution
        self.target_device = target_device
        self.dtype = dtype
        self.fill_placeholder = fill_placeholder
        self.H, self.W = target_resolution

    # ---------- 辅助：创建空占位符 ----------
    def _create_placeholder(self, factory_name: str, component_id: str) -> Dict[str, Any]:
        """
        为缺失的零件创建一个空占位符，防止总装环节 KeyError。
        占位符中包含 status: "VOID" 标志。
        """
        return {
            "factory": factory_name,
            "component_id": component_id,
            "status": "VOID",
            "base_normal": None,
            "normal_delta": None,
            "mask": None,
            "displacement": None,
            "color": None,
            "alpha": None,
            "joint_delta": None,
            "rig_map": None,
            # 其他可能的字段可根据需要扩展
        }

    # ---------- 核心：零件对齐 ----------
    def _align_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """
        对齐单个零件内的所有张量：
            - 分辨率 -> target_resolution（双线性插值）
            - 设备 -> target_device
            - 精度 -> dtype
        不修改非张量字段，不改变数值语义（仅几何变换）。
        """
        if component is None:
            return None
        # 如果是占位符，直接返回
        if component.get("status") == "VOID":
            return component

        # 定义需要检查的常见张量字段（可扩展）
        tensor_fields = [
            'base_normal', 'normal_delta', 'mask', 'displacement',
            'color', 'alpha', 'joint_delta', 'rig_map',
            'fluid_mask', 'fluid_normal_delta', 'refraction_data',
            'hole_mask', 'edge_debris', 'edge_normal_delta',
            'fabric_tensor', 'alpha_mask', 'detail_normal'
        ]

        for field in tensor_fields:
            if field not in component:
                continue
            tensor = component[field]
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue

            # 1. 精度转换
            if tensor.dtype != self.dtype:
                tensor = tensor.to(self.dtype)

            # 2. 设备迁移
            if tensor.device != self.target_device:
                tensor = tensor.to(self.target_device)

            # 3. 维度标准化：确保是 4D (B, C, H, W)
            original_ndim = tensor.ndim
            if original_ndim == 3:
                # 假设 (H, W, C) 或 (C, H, W)，统一为 (1, C, H, W)
                # 常见情况：法线图 (H,W,3) -> (1,3,H,W)
                if tensor.shape[-1] in (1, 2, 3, 4, 5):
                    tensor = tensor.permute(2, 0, 1)  # (C, H, W)
                tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            elif original_ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            # 如果已经是 4D 但通道在最后，转置
            elif original_ndim == 4 and tensor.shape[-1] in (1,2,3,4,5) and tensor.shape[1] not in (1,2,3,4,5):
                tensor = tensor.permute(0, 3, 1, 2)

            # 4. 分辨率对齐：强制到 (B, C, H_target, W_target)
            _, C, H_curr, W_curr = tensor.shape
            if (H_curr, W_curr) != (self.H, self.W):
                tensor = F.interpolate(
                    tensor,
                    size=(self.H, self.W),
                    mode='bilinear',
                    align_corners=False
                )

            # 5. 写回
            component[field] = tensor

        return component

    # ---------- 主入口：生成同步包 ----------
    def sync(self,
             components: List[Dict[str, Any]],
             expected_factories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        将零件列表同步为统一的同步包。

        参数:
            components: 工厂产出的零件列表（可能包含 None 或缺失项）
            expected_factories: 主模块期望出现的工厂名称列表，用于补全缺失零件

        返回:
            {
                "sync_timestamp": float,
                "global_resolution": (H, W),
                "target_device": device,
                "components": List[Dict],      # 对齐后的零件列表
                "mask_collective": torch.Tensor, # (1,1,H,W) 所有零件的有效区域并集
                "status": "OK" | "PARTIAL"
            }
        """
        start_time = time.time()

        # 1. 收集实际存在的工厂和 component_id
        existing_map = {}
        for comp in components:
            if comp is None:
                continue
            fac = comp.get('factory')
            cid = comp.get('component_id')
            if fac and cid:
                existing_map[fac] = comp

        # 2. 补全缺失零件（如果提供了 expected_factories）
        complete_components = []
        if self.fill_placeholder and expected_factories is not None:
            for fac in expected_factories:
                if fac in existing_map:
                    complete_components.append(existing_map[fac])
                else:
                    # 创建占位符
                    placeholder = self._create_placeholder(fac, f"void_{fac}")
                    complete_components.append(placeholder)
        else:
            complete_components = [c for c in components if c is not None]

        # 3. 对齐每个零件
        aligned = []
        for comp in complete_components:
            aligned_comp = self._align_component(comp)
            if aligned_comp is not None:
                aligned.append(aligned_comp)

        # 4. 生成 mask_collective：所有零件有效区域的并集（max 池化）
        # 收集每个零件中的 mask 字段（或根据其它字段生成有效区域）
        mask_list = []
        for comp in aligned:
            # 跳过占位符
            if comp.get("status") == "VOID":
                continue
            # 优先使用显式的 mask
            mask = comp.get("mask")
            if mask is not None and isinstance(mask, torch.Tensor):
                # 确保是 (B,1,H,W) 格式
                if mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 4 and mask.shape[1] != 1:
                    # 取平均或第一个通道作为遮罩
                    mask = mask[:, :1, :, :]
                mask_list.append(mask)
            else:
                # 如果没有显式 mask，但存在非 None 张量，认为该零件覆盖全区域
                # 简单起见：存在 base_normal 或 normal_delta 则视为全有效
                if comp.get("base_normal") is not None or comp.get("normal_delta") is not None:
                    full_mask = torch.ones(1, 1, self.H, self.W,
                                           device=self.target_device,
                                           dtype=self.dtype)
                    mask_list.append(full_mask)

        if mask_list:
            # 求所有 mask 的最大值（并集）
            mask_collective = torch.stack(mask_list, dim=0).max(dim=0)[0]
            # 确保形状为 (1,1,H,W)
            if mask_collective.ndim == 3:
                mask_collective = mask_collective.unsqueeze(0)
        else:
            mask_collective = torch.zeros(1, 1, self.H, self.W,
                                          device=self.target_device,
                                          dtype=self.dtype)

        # 5. 判断同步状态
        # 修复：expected_factories 为 None 时，无法判断，状态设为 "UNKNOWN"
        # 否则，统计非占位符的有效零件数量，只有所有期望工厂都有非占位符零件时才为 "OK"
        if expected_factories is None:
            status = "UNKNOWN"
        else:
            # 收集所有非占位符的工厂名
            active_factories = set()
            for comp in aligned:
                if comp.get("status") != "VOID":
                    fac = comp.get("factory")
                    if fac:
                        active_factories.add(fac)
            # 期望的工厂集合
            expected_set = set(expected_factories)
            if active_factories == expected_set:
                status = "OK"
            else:
                status = "PARTIAL"

        # 6. 封装同步包
        sync_package = {
            "sync_timestamp": start_time,
            "global_resolution": (self.H, self.W),
            "target_device": self.target_device,
            "components": aligned,
            "mask_collective": mask_collective,
            "status": status
        }
        return sync_package


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 模拟两个零件
    skin_comp = {
        "factory": "skin_factory",
        "component_id": "skin_abc",
        "base_normal": torch.rand(1, 3, 512, 512),
        "mask": torch.ones(1, 1, 512, 512),
    }
    fluid_comp = {
        "factory": "fluid_factory",
        "component_id": "fluid_xyz",
        "fluid_mask": torch.rand(1, 1, 256, 256),  # 分辨率不一致
        "fluid_normal_delta": torch.rand(1, 3, 256, 256),
    }

    # 桥接器配置
    bridge = P1NeuralSyncBridge(
        target_resolution=(1024, 1024),
        target_device=torch.device('cpu'),
        dtype=torch.float32,
        fill_placeholder=True
    )

    # 期望的工厂列表（主模块要求）
    expected = ["skin_factory", "joint_factory", "fluid_factory"]
    sync_pkg = bridge.sync([skin_comp, fluid_comp], expected_factories=expected)

    print("同步包状态:", sync_pkg["status"])
    print("全局分辨率:", sync_pkg["global_resolution"])
    print("零件数量:", len(sync_pkg["components"]))
    print("mask_collective shape:", sync_pkg["mask_collective"].shape)

    # 检查缺失的关节零件是否被补全
    for comp in sync_pkg["components"]:
        if comp.get("factory") == "joint_factory":
            print("关节零件状态:", comp.get("status", "ACTIVE"))