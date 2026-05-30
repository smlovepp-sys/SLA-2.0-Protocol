# p3_neural_sync_bridge.py
# P3 神经同步桥接器：非侵入式特征抓取与物理意识注入
# 实现潜空间拓扑映射、按需语义采样、注意力偏置注入、稀疏采样与跨帧缓存
# 修正：物理场下采样使用最大值池化 + 显著性增益，避免特征稀释；显存峰值对冲

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

class P3NeuralSyncBridge:
    """
    神经同步桥接器
    - 拓扑拦截：计算物理瓦片 → 潜空间特征点的精确映射
    - 按需语义采样：仅对 L0/L1 瓦片提取 UNet 中间层特征，识别材质语义
    - 注意力偏置注入：将物理应力/折痕编码为 Cross-Attention 的偏置权重
    - 稀疏张量化与跨帧缓存
    - 扩展：生成 1:8 逻辑拉伸引导场（供 P4 高精度还原）
    """

    def __init__(self, physical_grid_size: int = 32, latent_downscale_factor: int = 8, sync_gain: float = 1.5):
        self.phys_grid = physical_grid_size
        self.latent_downscale = latent_downscale_factor
        self.latent_tile_size = self.latent_downscale
        self.latent_grid = self.phys_grid * self.latent_tile_size
        self.sync_gain = sync_gain
        self.semantic_cache = {}
        self.frame_id = 0
        self.material_classifier = self._build_mock_classifier()
        self.attention_biases = {}

    def _build_mock_classifier(self):
        def classifier(features):
            return ["denim"] * features.shape[0]
        return classifier

    def sync_topology(self, latent_shape: Tuple[int, ...]) -> Dict[str, Any]:
        if len(latent_shape) == 4:
            B, C, H, W = latent_shape
        else:
            H, W = latent_shape[-2:]
        expected_h = self.latent_grid
        expected_w = self.latent_grid
        if H != expected_h or W != expected_w:
            print(f"[P3Bridge] Warning: latent shape ({H},{W}) != expected ({expected_h},{expected_w})")
        step = self.latent_tile_size
        mapping = {
            "physical_grid": self.phys_grid,
            "latent_grid": self.latent_grid,
            "tile_step": step,
            "latent_shape": (H, W),
            "valid": (H == expected_h and W == expected_w)
        }
        return mapping

    def _get_latent_region(self, tx: int, ty: int, latent_h: int, latent_w: int) -> Tuple[slice, slice]:
        step = self.latent_tile_size
        y_start = ty * step
        y_end = min(latent_h, (ty+1) * step)
        x_start = tx * step
        x_end = min(latent_w, (tx+1) * step)
        return slice(y_start, y_end), slice(x_start, x_end)

    def downsample_physics_to_latent(self, high_res_tensor: torch.Tensor) -> torch.Tensor:
        H_phys, W_phys = high_res_tensor.shape[-2:]
        assert H_phys % self.latent_downscale == 0 and W_phys % self.latent_downscale == 0, \
            f"物理场尺寸 ({H_phys},{W_phys}) 必须能被下采样因子 {self.latent_downscale} 整除"

        abs_tensor = torch.abs(high_res_tensor)
        pool_kernel = self.latent_downscale
        max_vals, indices = F.max_pool2d(
            abs_tensor,
            kernel_size=pool_kernel,
            stride=pool_kernel,
            return_indices=True
        )
        del abs_tensor  # 显存释放

        B, C, H_lat, W_lat = max_vals.shape
        flat_high = high_res_tensor.view(B, C, -1)
        flat_indices = indices.view(B, C, -1)
        gathered = torch.gather(flat_high, 2, flat_indices)
        latent_guidance = gathered.view(B, C, H_lat, W_lat)
        latent_guidance = latent_guidance * self.sync_gain
        return latent_guidance

    def sample_brain_features(self, unet_module: nn.Module, latent: torch.Tensor,
                              active_tiles: Dict[str, Set[Tuple[int, int]]]) -> Dict[str, Dict[Tuple[int, int], str]]:
        B, C, H, W = latent.shape
        results = defaultdict(dict)
        with torch.no_grad():
            for asset_id, tiles in active_tiles.items():
                for (tx, ty) in tiles:
                    y_slice, x_slice = self._get_latent_region(tx, ty, H, W)
                    region = latent[:, :, y_slice, x_slice]
                    feat = F.adaptive_avg_pool2d(region, (1, 1)).view(B, -1)
                    label = self.material_classifier(feat)[0]
                    results[asset_id][(tx, ty)] = label
                    cache_key = f"{asset_id}_{tx}_{ty}"
                    self.semantic_cache[cache_key] = (label, self.frame_id)
        return results

    def inject_consciousness(self, unet_module: nn.Module,
                             stress_field: Dict[str, Dict[Tuple[int, int], float]],
                             crease_field: Dict[str, Dict[Tuple[int, int], float]],
                             tear_field: Dict[str, Dict[Tuple[int, int], float]],
                             latent_shape: Tuple[int, int]) -> None:
        H, W = latent_shape
        bias_map = torch.zeros(1, H, W, dtype=torch.float32)
        for asset_id, stress_dict in stress_field.items():
            for (tx, ty), stress in stress_dict.items():
                y_slice, x_slice = self._get_latent_region(tx, ty, H, W)
                bias_map[:, y_slice, x_slice] += stress
        for asset_id, crease_dict in crease_field.items():
            for (tx, ty), crease in crease_dict.items():
                y_slice, x_slice = self._get_latent_region(tx, ty, H, W)
                bias_map[:, y_slice, x_slice] += crease * 0.5
        for asset_id, tear_dict in tear_field.items():
            for (tx, ty), tear in tear_dict.items():
                y_slice, x_slice = self._get_latent_region(tx, ty, H, W)
                bias_map[:, y_slice, x_slice] += tear * 0.8
        bias_map = torch.clamp(bias_map, 0.0, 1.0)
        bias_seq = bias_map.view(1, -1)
        if hasattr(unet_module, 'set_attention_bias'):
            unet_module.set_attention_bias(bias_seq)
        else:
            self.attention_biases['cross_attention'] = bias_seq
            unet_module._p3_attention_bias = bias_seq
        print(f"[P3Bridge] Injected attention bias with shape {bias_seq.shape}")

    def refine_physics_params(self, semantic_labels: Dict[str, Dict[Tuple[int, int], str]],
                              current_physics: Dict[str, Any]) -> Dict[str, Any]:
        material_factors = {
            "skin":   {"stiffness_factor": 0.8, "friction_factor": 0.6, "elasticity": 0.7},
            "denim":  {"stiffness_factor": 0.9, "friction_factor": 0.5, "elasticity": 0.4},
            "silk":   {"stiffness_factor": 0.3, "friction_factor": 0.2, "elasticity": 0.8},
            "leather":{"stiffness_factor": 1.0, "friction_factor": 0.9, "elasticity": 0.2},
            "metal":  {"stiffness_factor": 1.2, "friction_factor": 0.3, "elasticity": 0.1},
            "default":{"stiffness_factor": 0.7, "friction_factor": 0.5, "elasticity": 0.5},
        }
        adjusted = current_physics.copy()
        for asset_id, tiles in semantic_labels.items():
            for (tx, ty), label in tiles.items():
                mat = label.lower()
                factors = material_factors.get(mat, material_factors["default"])
                if asset_id in adjusted and (tx, ty) in adjusted[asset_id]:
                    phys = adjusted[asset_id][(tx, ty)]
                    phys["stiffness"] = min(1.0, phys.get("stiffness", 0.5) * factors["stiffness_factor"])
                    phys["friction"] = min(1.0, phys.get("friction", 0.5) * factors["friction_factor"])
                else:
                    if asset_id not in adjusted:
                        adjusted[asset_id] = {}
                    adjusted[asset_id][(tx, ty)] = {
                        "stiffness": factors["stiffness_factor"],
                        "friction": factors["friction_factor"],
                        "elasticity": factors["elasticity"],
                    }
        return adjusted

    def advance_frame(self):
        self.frame_id += 1
        max_age = 30
        to_delete = []
        for key, (_, frame) in self.semantic_cache.items():
            if self.frame_id - frame > max_age:
                to_delete.append(key)
        for key in to_delete:
            del self.semantic_cache[key]

    def get_cached_semantic(self, asset_id: str, tx: int, ty: int) -> Optional[str]:
        key = f"{asset_id}_{tx}_{ty}"
        if key in self.semantic_cache:
            label, _ = self.semantic_cache[key]
            return label
        return None

    def generate_expansion_guide(self, physical_payload: torch.Tensor) -> Dict[str, Any]:
        grad_x = torch.abs(physical_payload[:, :, :, 1:] - physical_payload[:, :, :, :-1])
        grad_y = torch.abs(physical_payload[:, :, 1:, :] - physical_payload[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
        gradient = torch.max(grad_x, grad_y)
        expansion_guide = F.interpolate(physical_payload, scale_factor=8, mode='bilinear', align_corners=False)
        boundary_sovereignty = (gradient > 0.1).float()
        return {
            "expansion_guide": expansion_guide,
            "boundary_sovereignty": boundary_sovereignty
        }


# 使用示例
if __name__ == "__main__":
    bridge = P3NeuralSyncBridge(physical_grid_size=32, latent_downscale_factor=8, sync_gain=1.5)
    phys_tensor = torch.randn(1, 4, 32, 32)
    latent_guide = bridge.downsample_physics_to_latent(phys_tensor)
    print(f"Downsampled latent guide shape: {latent_guide.shape}")
    latent_shape = (1, 16, 128, 128)
    mapping = bridge.sync_topology(latent_shape)
    print("Topology mapping:", mapping)