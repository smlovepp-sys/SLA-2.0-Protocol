# p1_tear_factory.py
# P1 撕裂零件厂：输出破坏与边缘零件（镂空掩模 + 边缘碎片 + 边缘法线修正增量）
# 修复版：移除 0.5 中性值残留，未激活时输出零张量

import hashlib
import threading
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F


class TearFactory:
    """
    撕裂零件厂：生产"破坏与边缘"零件。
    适配新的 (param_name, value) 令牌格式。
    无物理激活时输出全零张量。
    """

    def __init__(self):
        self.factory_name = "tear_factory"
        self._cache = {}
        self._cache_lock = threading.RLock()

    # ---------- 内部辅助 ----------
    def _extract_tear_params(self, audit_info: Dict[str, Any],
                             factory_tokens: List[Tuple[str, float]],
                             time_factor: float = 0.0) -> Dict[str, Any]:
        keyword_params = audit_info.get("keyword_params", {})
        intensity = keyword_params.get("tear_intensity", 0.0)

        for token in factory_tokens:
            param_name = token[0]
            value = token[1]
            if "_" in param_name:
                param_name = param_name.split("_")[-1]
            if param_name == "tear_intensity":
                intensity = value

        intensity = min(1.0, intensity * (1.0 + time_factor * 0.8))

        cx, cy = 0.5, 0.5
        for token, _ in factory_tokens:
            t = token.lower()
            if "left" in t:
                cx = 0.2
            elif "right" in t:
                cx = 0.8
            elif "top" in t:
                cy = 0.2
            elif "bottom" in t:
                cy = 0.8

        radius_ratio = 0.1 + intensity * 0.3
        edge_roughness = 0.5 + intensity * 0.7
        return {
            "intensity": intensity,
            "position": (cx, cy),
            "radius_ratio": radius_ratio,
            "edge_roughness": edge_roughness,
        }

    def _generate_hole_mask(self, params: Dict[str, Any],
                            target_shape: Tuple[int, int, int, int],
                            device: str) -> torch.Tensor:
        B, C, H, W = target_shape
        intensity = params["intensity"]
        if intensity <= 0:
            return torch.zeros(B, 1, H, W, device=device)

        cx, cy = params["position"]
        radius_ratio = params["radius_ratio"]
        radius = int(radius_ratio * min(H, W))
        px, py = int(cx * W), int(cy * H)

        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device),
                              indexing='ij')
        dist = torch.sqrt((x - px) ** 2 + (y - py) ** 2)

        inner_radius = radius * (1 - 0.2 * intensity)
        outer_radius = radius

        mask = torch.zeros(B, 1, H, W, device=device)
        mask[0, 0] = (dist <= inner_radius).float()
        transition = (dist > inner_radius) & (dist <= outer_radius)
        if transition.any():
            alpha = 1.0 - (dist[transition] - inner_radius) / (outer_radius - inner_radius)
            mask[0, 0, transition] = alpha

        mask = mask * intensity
        kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
        mask = F.conv2d(mask, kernel, padding=1)
        return torch.clamp(mask, 0.0, 1.0)

    def _generate_edge_debris(self, params: Dict[str, Any],
                              hole_mask: torch.Tensor,
                              target_shape: Tuple[int, int, int, int],
                              device: str) -> torch.Tensor:
        B, C, H, W = target_shape
        intensity = params["intensity"]
        edge_roughness = params["edge_roughness"]

        if intensity <= 0:
            return torch.zeros(B, 3, H, W, device=device)

        kernel = torch.ones(1, 1, 3, 3, device=device)
        dilated = F.conv2d(hole_mask, kernel, padding=1) > 0
        eroded = F.conv2d(hole_mask, kernel, padding=1) > 0.99
        edge = (dilated.float() - eroded.float()).clamp(0, 1)

        debris = torch.rand(B, 3, H, W, device=device) * 0.5
        debris[:, 0, :, :] += 0.3
        debris[:, 1, :, :] *= 0.3
        debris[:, 2, :, :] *= 0.2

        noise = torch.rand(B, 1, H, W, device=device) < (edge_roughness * 0.3)
        debris = debris * noise.float() * edge * intensity
        return debris

    def _generate_edge_normal_delta(self, params: Dict[str, Any],
                                    hole_mask: torch.Tensor,
                                    target_shape: Tuple[int, int, int, int],
                                    device: str,
                                    external_normal_kernel: Any = None) -> torch.Tensor:
        B, C, H, W = target_shape
        intensity = params["intensity"]
        if intensity <= 0:
            return torch.zeros(B, 3, H, W, device=device)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        grad_x = F.conv2d(hole_mask, sobel_x, padding=1)
        grad_y = F.conv2d(hole_mask, sobel_y, padding=1)

        if external_normal_kernel is not None and external_normal_kernel.is_available():
            kernel_input = torch.cat([grad_x, grad_y], dim=1)
            enhanced = external_normal_kernel.apply(kernel_input, target_channels=3)
            if enhanced is not None:
                if enhanced.min() < 0:
                    enhanced = (enhanced + 1.0) / 2.0
                normal_delta = torch.clamp(enhanced, 0.0, 1.0)
                kernel = torch.ones(1, 1, 3, 3, device=device)
                edge_region = (F.conv2d(hole_mask, kernel, padding=1) > 0).float()
                edge_region = edge_region - hole_mask
                edge_region = (edge_region > 0).float()
                edge_region_3ch = edge_region.repeat(1, 3, 1, 1)
                neutral = torch.zeros(B, 3, H, W, device=device)
                return neutral * (1 - edge_region_3ch) + normal_delta * edge_region_3ch

        edge_roughness = params["edge_roughness"]
        strength = intensity * edge_roughness

        normal = torch.zeros(B, 3, H, W, device=device)
        normal[:, 0, :, :] = grad_x[:, 0, :, :] * strength
        normal[:, 1, :, :] = grad_y[:, 0, :, :] * strength
        normal[:, 2, :, :] = 1.0

        norm_len = torch.sqrt((normal ** 2).sum(dim=1, keepdim=True) + 1e-8)
        normal = normal / norm_len
        normal_delta = (normal + 1.0) / 2.0

        kernel = torch.ones(1, 1, 3, 3, device=device)
        edge_region = (F.conv2d(hole_mask, kernel, padding=1) > 0).float()
        edge_region = edge_region - hole_mask
        edge_region = (edge_region > 0).float()
        edge_region_3ch = edge_region.repeat(1, 3, 1, 1)

        # 边缘区域使用动态法线，非边缘区域为零
        neutral = torch.zeros(B, 3, H, W, device=device)
        normal_delta = neutral * (1 - edge_region_3ch) + normal_delta * edge_region_3ch
        return normal_delta

    def _generate_alignment_meta(self, ref_resolution: Tuple[int, int],
                                 tear_intensity: float) -> Dict[str, Any]:
        return {
            "reference_resolution": ref_resolution,
            "tear_intensity": tear_intensity,
            "operation": "boolean_subtract",
            "edge_blend_width": 3,
        }

    def _align_to_target(self, tensor: torch.Tensor,
                         target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        Bt, Ct, Ht, Wt = target_shape
        if tensor.ndim == 3:
            if tensor.shape[-1] in [1, 3]:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[-1] in [1, 3] and tensor.shape[1] not in [1, 3]:
                tensor = tensor.permute(0, 3, 1, 2)
        if tensor.shape[1] != Ct:
            if tensor.shape[1] < Ct:
                repeats = [1, Ct // tensor.shape[1] + 1, 1, 1]
                tensor = tensor.repeat(*repeats)[:, :Ct, :, :]
            else:
                tensor = tensor[:, :Ct, :, :]
        if tensor.shape[2] != Ht or tensor.shape[3] != Wt:
            tensor = F.interpolate(tensor, size=(Ht, Wt), mode='bilinear', align_corners=False)
        if tensor.shape[0] != Bt:
            tensor = tensor[:Bt] if tensor.shape[0] > Bt else tensor.repeat(Bt, 1, 1, 1)
        return tensor

    def _expand_normal_delta_to_target(self, normal_delta: torch.Tensor, target_channels: int, add_noise: bool = True) -> torch.Tensor:
        if target_channels <= 3:
            return normal_delta
        B, C, H, W = normal_delta.shape
        repeats = target_channels // 3 + 1
        expanded = normal_delta.repeat(1, repeats, 1, 1)[:, :target_channels, :, :]
        if add_noise:
            noise = torch.randn_like(expanded) * 0.1
            expanded = expanded + noise
        return torch.clamp(expanded, 0.0, 1.0)

    # ---------- 主入口 ----------
    def produce(self,
                factory_tokens: List[Tuple[str, float]],
                audit_info: Dict[str, Any] = None,
                external_normal_kernel: Any = None,
                target_info: Dict[str, Any] = None,
                global_resolution: Tuple[int, int] = (512, 512),
                target_batch: int = 1,
                device: str = 'cpu',
                target_channels: int = 4,
                time_factor: float = 0.0) -> Dict[str, Any]:
        if target_info is None:
            target_info = {"thickness": 0.5, "material_type": "skin"}
        if audit_info is None:
            audit_info = {}

        H, W = global_resolution

        # 如果没有物理激活，返回全零张量
        if len(factory_tokens) == 0:
            full_tensor = torch.zeros(target_batch, target_channels, H, W, device=device)
            return {
                "component_id": hashlib.md5(f"{self.factory_name}_neutral_{target_batch}_{H}_{W}".encode()).hexdigest()[:12],
                "tensor": full_tensor,
                "hole_mask": torch.zeros(target_batch, 1, H, W, device=device),
                "edge_debris": torch.zeros(target_batch, 3, H, W, device=device),
                "edge_normal_delta": torch.zeros(target_batch, 3, H, W, device=device),
                "alignment_meta": self._generate_alignment_meta(global_resolution, 0.0),
                "factory": self.factory_name,
            }

        tear_params = self._extract_tear_params(audit_info, factory_tokens, time_factor)
        intensity = tear_params["intensity"]

        mask_shape = (target_batch, 1, H, W)
        debris_shape = (target_batch, 3, H, W)
        normal_base_shape = (target_batch, 3, H, W)

        hole_mask = self._generate_hole_mask(tear_params, mask_shape, device)
        hole_mask = self._align_to_target(hole_mask, mask_shape)

        edge_debris = self._generate_edge_debris(tear_params, hole_mask, debris_shape, device)
        edge_debris = self._align_to_target(edge_debris, debris_shape)

        edge_normal_delta = self._generate_edge_normal_delta(tear_params, hole_mask, normal_base_shape, device,
                                                             external_normal_kernel=external_normal_kernel)
        edge_normal_delta = self._align_to_target(edge_normal_delta, normal_base_shape)

        if target_channels <= 3:
            final_tensor = edge_normal_delta[:, :target_channels, :, :]
        elif target_channels == 4:
            final_tensor = torch.cat([edge_normal_delta, hole_mask], dim=1)
        else:
            final_tensor = torch.cat([edge_normal_delta, hole_mask], dim=1)
            remaining = target_channels - 4
            if remaining > 0:
                extended = self._expand_normal_delta_to_target(edge_normal_delta, remaining, add_noise=True)
                final_tensor = torch.cat([final_tensor, extended], dim=1)

        alignment_meta = self._generate_alignment_meta(global_resolution, intensity)
        component_id = hashlib.md5(
            f"{self.factory_name}_{intensity:.3f}_{global_resolution}_{target_channels}_{time_factor:.2f}".encode()
        ).hexdigest()[:12]

        token_weights = [w for _, w in factory_tokens]
        cache_key = hashlib.md5(
            (str(token_weights) + str(global_resolution) + str(target_batch) +
             device + str(target_channels) + str(time_factor)).encode()
        ).hexdigest()
        with self._cache_lock:
            self._cache[cache_key] = final_tensor

        component = {
            "component_id": component_id,
            "tensor": final_tensor,
            "hole_mask": hole_mask,
            "edge_debris": edge_debris,
            "edge_normal_delta": edge_normal_delta,
            "alignment_meta": alignment_meta,
            "factory": self.factory_name,
        }

        return component