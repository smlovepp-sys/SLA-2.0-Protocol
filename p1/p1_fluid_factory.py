# p1_fluid_factory.py
# P1 流体零件厂：输出附着式流体零件（遮罩 + 法线修正增量 + 折射数据）
# 统一 **kwargs 接口版

import hashlib
import threading
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F

class FluidFactory:
    """
    流体零件厂：生产附着式流体特效组件。
    适配统一 **kwargs 接口，无物理激活时输出全零张量。
    """

    FLUID_PRESETS = {
        "L":  {"viscosity": 0.5, "ior": 1.35, "diffusion_rate": 0.6, "specular": 0.5},
        "Lh": {"viscosity": 0.1, "ior": 1.33, "diffusion_rate": 0.3, "specular": 0.7},
        "Ls": {"viscosity": 0.6, "ior": 1.35, "diffusion_rate": 0.7, "specular": 0.3},
        "Lm": {"viscosity": 0.0, "ior": 1.31, "diffusion_rate": 0.2, "specular": 0.9},
        "Lv": {"viscosity": 0.9, "ior": 1.45, "diffusion_rate": 0.9, "specular": 0.2},
        "fluid": {"viscosity": 0.4, "ior": 1.33, "diffusion_rate": 0.5, "specular": 0.6}
    }

    def __init__(self):
        self.factory_name = "fluid_factory"
        self._cache = {}
        self._cache_lock = threading.RLock()

    # ---------- 内部辅助计算 ----------
    def _resolve_fluid_props(self, factory_tokens: List) -> Tuple[str, float]:
        """解析流体类型与演化率，令牌兼容新旧格式"""
        sub_type = "fluid"
        evolution_rate = 0.0  # 默认无令牌时将被跳过

        for token in factory_tokens:
            if isinstance(token, (list, tuple)) and len(token) == 2:
                if isinstance(token[1], dict):
                    # 新格式: (object_name, {"wetness": 0.8, ...})
                    params = token[1]
                    wetness = params.get("wetness", 0.0)
                    friction = params.get("friction", 0.5)
                    evolution_rate = max(evolution_rate, max(0.0, min(1.0, wetness)))
                    evolution_rate += (1.0 - friction) * 0.2
                    # 尝试从物体名匹配流体类型
                    obj_name = token[0].lower()
                    for k in self.FLUID_PRESETS:
                        if k.lower() in obj_name:
                            sub_type = k
                            break
                else:
                    # 旧格式: ("name_param", value)
                    param_name = token[0]
                    value = token[1]
                    if "_" in param_name:
                        param_name = param_name.split("_")[-1]
                    if param_name == "wetness":
                        evolution_rate = max(0.0, min(1.0, value))
                    elif param_name == "friction":
                        evolution_rate += (1.0 - value) * 0.2
                    # 简单匹配
                    t = param_name.lower()
                    for k in self.FLUID_PRESETS:
                        if k.lower() in t:
                            sub_type = k
                            break

        evolution_rate = max(0.0, min(1.0, evolution_rate))
        return sub_type, evolution_rate

    def _generate_fluid_mask(self, batch: int, H: int, W: int, evo_rate: float, device: torch.device, time_factor: float) -> torch.Tensor:
        if evo_rate <= 0.01:
            return torch.zeros(batch, 1, H, W, device=device)

        yy, xx = torch.meshgrid(
            torch.linspace(-2.0, 2.0, H, device=device, dtype=torch.float32),
            torch.linspace(-2.0, 2.0, W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        
        wave = torch.sin(xx * 2.0 + time_factor) * torch.cos(yy * 2.0 - time_factor * 0.5)
        mask = (wave + 1.0) / 2.0
        
        threshold = 1.0 - evo_rate
        fluid_mask = torch.where(mask > threshold, mask, torch.zeros_like(mask))
        fluid_mask = torch.clamp(fluid_mask * 1.5, 0.0, 1.0)

        return fluid_mask.unsqueeze(0).repeat(batch, 1, 1).unsqueeze(1)

    def _generate_fluid_normal(self, fluid_mask: torch.Tensor, batch: int, H: int, W: int, device: torch.device, external_normal_kernel: Optional[Any] = None) -> torch.Tensor:
        if external_normal_kernel and getattr(external_normal_kernel, "kernel", None) is not None:
            ext_out = external_normal_kernel.infer(fluid_mask, target_channels=3)
            if ext_out is not None:
                return ext_out

        dx = fluid_mask[:, 0:1, :, :-1] - fluid_mask[:, 0:1, :, 1:]
        dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
        
        dy = fluid_mask[:, 0:1, :-1, :] - fluid_mask[:, 0:1, 1:, :]
        dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

        normal_x = torch.clamp(dx * 4.0 + 0.5, 0.0, 1.0)
        normal_y = torch.clamp(dy * 4.0 + 0.5, 0.0, 1.0)
        normal_z = torch.clamp(1.0 - fluid_mask * 0.5, 0.5, 1.0)

        return torch.cat([normal_x, normal_y, normal_z], dim=1)

    def _generate_refraction(self, fluid_mask: torch.Tensor, ior: float) -> torch.Tensor:
        dx = fluid_mask[:, 0:1, :, :-1] - fluid_mask[:, 0:1, :, 1:]
        dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
        
        dy = fluid_mask[:, 0:1, :-1, :] - fluid_mask[:, 0:1, 1:, :]
        dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

        refract_scale = (ior - 1.0) * 0.5
        refract_x = dx * refract_scale
        refract_y = dy * refract_scale

        return torch.cat([refract_x, refract_y], dim=1)

    def _expand_normal_delta_to_target(self, normal_delta: torch.Tensor, remaining_channels: int) -> torch.Tensor:
        B = normal_delta.shape[0]
        repeats = (remaining_channels + 2) // 3
        extended = normal_delta.repeat(1, repeats, 1, 1)
        
        noise = torch.randn((B, remaining_channels, normal_delta.shape[-2], normal_delta.shape[-1]), 
                            device=normal_delta.device, dtype=normal_delta.dtype) * 0.01
        return extended[:, :remaining_channels, :, :] + noise

    def _align_to_target(self, tensor: torch.Tensor, target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        if tensor.shape == target_shape:
            return tensor
        if tensor.shape[-2:] != target_shape[-2:]:
            tensor = F.interpolate(tensor, size=target_shape[-2:], mode="bilinear", align_corners=False)
        if tensor.shape[0] != target_shape[0]:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(target_shape[0], 1, 1, 1)
            else:
                tensor = tensor[:target_shape[0], :, :, :]
        return tensor

    # ---------- 统一入口：produce(**kwargs) ----------
    def produce(self, **kwargs) -> Dict[str, Any]:
        """
        流体生产，所有参数均从关键字参数字典提取：
        - factory_tokens: List
        - global_resolution: Tuple[int, int]  默认 (64, 64)
        - target_channels: int                默认 12
        - target_batch: int                   默认 1
        - device: torch.device                默认 cpu
        - external_normal_kernel: Optional    默认 None
        - time_factor: float                  默认 0.0
        """
        factory_tokens = kwargs.get("factory_tokens", [])
        global_resolution = kwargs.get("global_resolution", (64, 64))
        target_channels = kwargs.get("target_channels", 12)
        target_batch = kwargs.get("target_batch", 1)
        device = kwargs.get("device", torch.device("cpu"))
        external_normal_kernel = kwargs.get("external_normal_kernel", None)
        time_factor = kwargs.get("time_factor", 0.0)

        H, W = global_resolution

        # 无令牌 → 零张量
        if len(factory_tokens) == 0:
            full_tensor = torch.zeros(target_batch, target_channels, H, W, device=device)
            return {
                "component_id": hashlib.md5(f"{self.factory_name}_neutral_{target_batch}_{H}_{W}".encode()).hexdigest()[:12],
                "tensor": full_tensor,
                "fluid_mask": torch.zeros(target_batch, 1, H, W, device=device),
                "fluid_normal_delta": torch.zeros(target_batch, 3, H, W, device=device),
                "refraction_data": torch.zeros(target_batch, 2, H, W, device=device),
                "factory": self.factory_name,
                "layer_depth": 3
            }

        sub_type, effective_evolution = self._resolve_fluid_props(factory_tokens)
        ior = self.FLUID_PRESETS[sub_type]["ior"]

        token_hash = hashlib.md5(str(factory_tokens).encode()).hexdigest()[:8]
        cache_key = f"{target_batch}_{target_channels}_{H}_{W}_{token_hash}_{device}_{time_factor:.2f}"

        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        fluid_mask = self._generate_fluid_mask(target_batch, H, W, effective_evolution, device, time_factor)
        fluid_normal_delta = self._generate_fluid_normal(
            fluid_mask, target_batch, H, W, device, external_normal_kernel=external_normal_kernel
        )
        refraction_data = self._generate_refraction(fluid_mask, ior)

        front_package = torch.cat([fluid_normal_delta, fluid_mask], dim=1)

        if target_channels <= 4:
            final_tensor = front_package[:, :target_channels, :, :]
        else:
            remaining = target_channels - 4
            extended_channels = self._expand_normal_delta_to_target(fluid_normal_delta, remaining)
            final_tensor = torch.cat([front_package, extended_channels], dim=1)

        target_shape = (target_batch, target_channels, H, W)
        final_tensor = self._align_to_target(final_tensor, target_shape)

        component_id = hashlib.md5(
            f"{self.factory_name}_{sub_type}_{target_batch}_{H}_{W}".encode()
        ).hexdigest()[:12]

        component = {
            "component_id": component_id,
            "tensor": final_tensor,
            "fluid_mask": fluid_mask,
            "fluid_normal_delta": fluid_normal_delta,
            "refraction_data": refraction_data,
            "factory": self.factory_name,
            "layer_depth": 3
        }

        with self._cache_lock:
            self._cache[cache_key] = component

        return component