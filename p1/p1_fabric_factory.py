# p1_fabric_factory.py
# P1 织物零件厂：输出织物外壳零件（外观 + 遮罩 + 细节法线）- 统一 **kwargs 接口

import hashlib
import threading
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F

class FabricFactory:
    """
    织物零件厂：生产织物外壳及多层物理叠加图层。
    完美支持多批次 (target_batch) 动态容器焊接。
    统一接口：所有参数通过关键字传递。
    """

    DEFAULT_MATERIALS = {
        "silk":    {"roughness": 0.15, "normal_strength": 0.2, "weave_density": 0.9},
        "satin":   {"roughness": 0.12, "normal_strength": 0.2, "weave_density": 0.95},
        "denim":   {"roughness": 0.75, "normal_strength": 0.6, "weave_density": 0.7},
        "canvas":  {"roughness": 0.8,  "normal_strength": 0.7, "weave_density": 0.65},
        "leather": {"roughness": 0.45, "normal_strength": 0.5, "weave_density": 0.5},
        "cotton":  {"roughness": 0.6,  "normal_strength": 0.3, "weave_density": 0.8},
        "fabric":  {"roughness": 0.5,  "normal_strength": 0.4, "weave_density": 0.75}
    }

    def __init__(self):
        self.factory_name = "fabric_factory"
        self._cache = {}
        self._cache_lock = threading.RLock()

    # ---------- 内部辅助计算 ----------
    def _resolve_material_props(self, factory_tokens: List) -> Tuple[str, float, float]:
        """根据工厂令牌（兼容新旧格式）解析材质类型、粗糙度及覆盖率"""
        mat_type = "fabric"
        stiffness = 0.5
        friction = 0.5
        elasticity = 0.5
        wetness = 0.0
        # tear = 0.0
        # tension = 0.0

        for token in factory_tokens:
            if isinstance(token, (list, tuple)) and len(token) == 2:
                if isinstance(token[1], dict):
                    # 新格式：(object_name, {"stiffness": 0.2, ...})
                    params = token[1]
                    stiffness = params.get("stiffness", stiffness)
                    friction = params.get("friction", friction)
                    elasticity = params.get("elasticity", elasticity)
                    wetness = params.get("wetness", wetness)
                    # 尝试从物体名中提取材质
                    obj_name = token[0].lower()
                    for mat in self.DEFAULT_MATERIALS:
                        if mat in obj_name:
                            mat_type = mat
                            break
                else:
                    # 旧格式：("obj_name_param", value)
                    param_name = token[0]
                    value = token[1]
                    if "_" in param_name:
                        parts = param_name.split("_")
                        pname = parts[-1]
                        # 尝试匹配材质
                        for part in parts[:-1]:
                            for mat in self.DEFAULT_MATERIALS:
                                if mat in part.lower():
                                    mat_type = mat
                                    break
                        if pname == "stiffness":
                            stiffness = value
                        elif pname == "friction":
                            friction = value
                        elif pname == "elasticity":
                            elasticity = value
                        elif pname == "wetness":
                            wetness = value
                        # 可扩展 tear, tension

        # 粗糙度 = 摩擦 * 0.6 + 刚度 * 0.4
        roughness = max(0.05, min(0.95, friction * 0.6 + stiffness * 0.4))
        # 覆盖率 = 弹性 * 0.7 + 0.3 - 湿度 * 0.3
        coverage = max(0.0, min(1.0, elasticity * 0.7 + 0.3 - wetness * 0.3))

        return mat_type, roughness, coverage

    def _generate_alpha_mask(self, batch: int, H: int, W: int, coverage: float, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        base_mask = (xx + yy + 2.0) / 4.0
        base_mask = torch.clamp(base_mask * coverage * 1.5, 0.0, 1.0)
        return base_mask.unsqueeze(0).repeat(batch, 1, 1).unsqueeze(1)

    def _generate_detail_normal(self, 
                                 mat_type: str, 
                                 alpha_mask: torch.Tensor, 
                                 batch: int, 
                                 H: int, 
                                 W: int, 
                                 device: torch.device, 
                                 time_factor: float, 
                                 external_normal_kernel: Optional[Any] = None) -> torch.Tensor:
        if external_normal_kernel and getattr(external_normal_kernel, "kernel", None) is not None:
            ext_out = external_normal_kernel.infer(alpha_mask, target_channels=3)
            if ext_out is not None:
                return ext_out

        props = self.DEFAULT_MATERIALS[mat_type]
        density = props["weave_density"]

        yy, xx = torch.meshgrid(
            torch.linspace(0, 50.0 * density, H, device=device, dtype=torch.float32),
            torch.linspace(0, 50.0 * density, W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        
        wave = torch.sin(xx + time_factor) * torch.cos(yy - time_factor)
        normal_x = torch.clamp(wave * 0.2 + 0.5, 0.0, 1.0)
        normal_y = torch.clamp(wave * 0.2 + 0.5, 0.0, 1.0)
        normal_z = torch.ones_like(wave)

        detail_normal = torch.cat([
            normal_x.unsqueeze(0).unsqueeze(1),
            normal_y.unsqueeze(0).unsqueeze(1),
            normal_z.unsqueeze(0).unsqueeze(1)
        ], dim=1)

        return detail_normal.repeat(batch, 1, 1, 1) * alpha_mask

    def _expand_detail_normal_to_target(self, detail_normal: torch.Tensor, remaining_channels: int) -> torch.Tensor:
        repeats = (remaining_channels + 2) // 3
        extended = detail_normal.repeat(1, repeats, 1, 1)
        return extended[:, :remaining_channels, :, :]

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
        织物生产，所需参数均从关键字参数字典中提取：
        - factory_tokens: List
        - global_resolution: Tuple[int, int]  默认 (64, 64)
        - target_channels: int                默认 8
        - target_batch: int                   默认 1
        - device: torch.device                默认 cpu
        - external_normal_kernel: Optional    默认 None
        - time_factor: float                  默认 0.0
        """
        factory_tokens = kwargs.get("factory_tokens", [])
        global_resolution = kwargs.get("global_resolution", (64, 64))
        target_channels = kwargs.get("target_channels", 8)
        target_batch = kwargs.get("target_batch", 1)
        device = kwargs.get("device", torch.device("cpu"))
        external_normal_kernel = kwargs.get("external_normal_kernel", None)
        time_factor = kwargs.get("time_factor", 0.0)

        H, W = global_resolution

        mat_type, roughness, coverage = self._resolve_material_props(factory_tokens)

        token_hash = hashlib.md5(str(factory_tokens).encode()).hexdigest()[:8]
        cache_key = f"{target_batch}_{target_channels}_{H}_{W}_{token_hash}_{device}_{time_factor:.2f}"

        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        alpha_mask = self._generate_alpha_mask(target_batch, H, W, coverage, device)
        detail_normal = self._generate_detail_normal(
            mat_type, alpha_mask, target_batch, H, W, device, time_factor,
            external_normal_kernel=external_normal_kernel
        )

        roughness_layer = torch.ones((target_batch, 1, H, W), device=device, dtype=torch.float32) * roughness
        fabric_base = torch.cat([roughness_layer, alpha_mask, detail_normal], dim=1)

        if target_channels <= 5:
            final_tensor = fabric_base[:, :target_channels, :, :]
        else:
            remaining = target_channels - 5
            extended_normals = self._expand_detail_normal_to_target(detail_normal, remaining)
            final_tensor = torch.cat([fabric_base, extended_normals], dim=1)

        target_shape = (target_batch, target_channels, H, W)
        final_tensor = self._align_to_target(final_tensor, target_shape)

        component_id = hashlib.md5(
            f"{self.factory_name}_{mat_type}_{target_batch}_{H}_{W}".encode()
        ).hexdigest()[:12]

        component = {
            "component_id": component_id,
            "tensor": final_tensor,
            "alpha_mask": alpha_mask,
            "detail_normal": detail_normal,
            "factory": self.factory_name,
            "layer_depth": 2
        }

        with self._cache_lock:
            self._cache[cache_key] = component

        return component