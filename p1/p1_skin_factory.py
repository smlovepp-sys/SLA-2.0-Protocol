# p1_skin_factory.py
# P1 皮肤零件厂：生产基础法线底盘，并填充额外的物理特征通道。
# V2.4 统一接口版：produce(**kwargs)，消除签名特异性

import hashlib
import threading
import functools
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F


class SkinFactory:
    """
    皮肤零件厂：生产基础法线底盘，并填充额外的物理特征通道。
    """
    def __init__(self, config: Dict[str, Any] = None, cache_size_limit: int = 128):
        self.factory_name = "skin_factory"
        self._cache = {}
        self._cache_lock = threading.RLock()
        self.cache_size_limit = cache_size_limit

        self.config = config or {
            "wetness_gain": 0.05,
            "smoothness_gain": 0.02,
            "warmth_gain": 0.03,
            "smoothness_min": 0.1,
            "smoothness_max": 0.95,
            "min_resolution": 16
        }

    # -------------------------------------------------------------------------
    # 核心装饰器（适配统一 **kwargs 接口）
    # -------------------------------------------------------------------------
    def _cache_and_protect(func):
        @functools.wraps(func)
        def wrapper(self, **kwargs):
            # 从 kwargs 中提取缓存键所需的关键参数
            factory_tokens = kwargs.get("factory_tokens", [])
            global_resolution = kwargs.get("global_resolution", (512, 512))
            target_batch = kwargs.get("target_batch", 1)
            target_channels = kwargs.get("target_channels", 16)
            device = kwargs.get("device", torch.device('cpu'))
            time_factor = kwargs.get("time_factor", 0.0)

            H, W = global_resolution
            min_res = self.config.get("min_resolution", 16)
            safe_H, safe_W = max(min_res, H), max(min_res, W)

            sorted_tokens = self._normalize_tokens(factory_tokens)
            token_hash = hashlib.md5(str(sorted_tokens).encode()).hexdigest()[:8]
            cache_key = f"{target_batch}_{target_channels}_{safe_H}_{safe_W}_{token_hash}_{device}_{time_factor:.2f}"

            with self._cache_lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

                if len(self._cache) > self.cache_size_limit:
                    self._cache = {}

                try:
                    # 原函数现在也接受 **kwargs，直接解包传递
                    result = func(self, **kwargs)
                    self._cache[cache_key] = result
                    return result
                except Exception as e:
                    print(f"[SkinFactory] ❌ 组件生成失败，触发自愈: {str(e)}")
                    return self._get_default_component(target_batch, target_channels, safe_H, safe_W, device)
        return wrapper

    # -------------------------------------------------------------------------
    # 统一入口：produce(**kwargs)
    # -------------------------------------------------------------------------
    @_cache_and_protect
    def produce(self, **kwargs):
        """
        所有参数均通过关键字传递，按需提取：
        - global_physics : Dict[str, float]  全局物理参数
        - factory_tokens : List              本工厂的令牌列表
        - global_resolution : Tuple[int, int]
        - target_channels : int (默认16)
        - target_batch : int (默认1)
        - device : torch.device
        - external_normal_kernel : Optional
        - time_factor : float (默认0.0)
        """
        # 提取所有需要的参数
        global_physics = kwargs.get("global_physics", {})
        factory_tokens = kwargs.get("factory_tokens", [])
        global_resolution = kwargs.get("global_resolution", (512, 512))
        target_channels = kwargs.get("target_channels", 16)
        target_batch = kwargs.get("target_batch", 1)
        device = kwargs.get("device", torch.device('cpu'))
        external_normal_kernel = kwargs.get("external_normal_kernel", None)
        time_factor = kwargs.get("time_factor", 0.0)

        H, W = global_resolution

        # 计算本工厂专属的物理参数（由 global_physics 和 tokens 融合）
        params = self._compute_physical_params(global_physics, factory_tokens)

        # 判断是否有物理激活
        has_physics = len(factory_tokens) > 0

        wetness = max(0.0, min(1.0, params["wetness"] * (1.0 + time_factor * 0.1)))
        smoothness = max(0.0, min(1.0, params["smoothness"] * (1.0 - time_factor * 0.05)))
        warmth = params["warmth"]

        # 无令牌时输出零张量
        if not has_physics:
            full_tensor = torch.zeros(target_batch, target_channels, H, W, device=device)
            return {
                "component_id": hashlib.md5(f"{self.factory_name}_neutral_{H}_{W}".encode()).hexdigest()[:12],
                "tensor": full_tensor,
                "base_normal": torch.zeros(target_batch, 3, H, W, device=device),
                "mask": torch.zeros(target_batch, 1, H, W, device=device),
                "factory": self.factory_name,
                "layer_depth": 0
            }

        # 生成基础法线底盘
        normal_3ch = self._generate_base_normal(wetness, smoothness, target_batch, H, W, device, external_normal_kernel)

        # 填充额外通道
        if target_channels > 3:
            extra = self._generate_extra_channels(wetness, smoothness, warmth, target_batch, target_channels - 3, H, W, device)
            full_tensor = torch.cat([normal_3ch, extra], dim=1)
        else:
            full_tensor = normal_3ch[:, :target_channels, :, :]

        full_tensor = self._align_to_target(full_tensor, (target_batch, target_channels, H, W))

        return {
            "component_id": hashlib.md5(f"{self.factory_name}_{wetness:.3f}_{smoothness:.3f}_{H}_{W}".encode()).hexdigest()[:12],
            "tensor": full_tensor,
            "base_normal": normal_3ch,
            "mask": torch.ones((target_batch, 1, H, W), device=device, dtype=torch.float32),
            "factory": self.factory_name,
            "layer_depth": 0
        }

    @staticmethod
    def _normalize_tokens(tokens):
        return sorted(tokens, key=lambda x: x[0])

    def _compute_physical_params(self, global_physics: Dict[str, float], factory_tokens: list):
        """
        从 global_physics 字典和本厂令牌中融合出本厂所需的三个派生参数：
        wetness, smoothness, warmth。
        """
        # 初始值来自全局物理参数（若未提供则使用中性值）
        stiffness = global_physics.get("stiffness", 0.5)
        friction = global_physics.get("friction", 0.5)
        elasticity = global_physics.get("elasticity", 0.5)
        wetness = global_physics.get("wetness", 0.0)
        tear = global_physics.get("tear_intensity", 0.0)
        tension = global_physics.get("tension", 0.0)

        # 令牌可以覆盖部分参数（与 master_manager 传来的格式兼容）
        for token in factory_tokens:
            # token 形如 (obj_name, {"stiffness": 0.2, ...}) 或 ("obj_stiffness", 0.2)
            if isinstance(token, (list, tuple)) and len(token) == 2:
                if isinstance(token[1], dict):
                    # 新版结构
                    for k, v in token[1].items():
                        if k == "stiffness":
                            stiffness = float(v)
                        elif k == "friction":
                            friction = float(v)
                        elif k == "elasticity":
                            elasticity = float(v)
                        elif k == "wetness":
                            wetness = float(v)
                        elif k == "tear_intensity":
                            tear = float(v)
                        elif k == "tension":
                            tension = float(v)
                else:
                    # 旧版字符串结构：token[0] 是 "obj_param", token[1] 是值
                    param_name = token[0]
                    value = token[1]
                    if "_" in param_name:
                        param_name = param_name.split("_")[-1]
                    if param_name == "stiffness":
                        stiffness = value
                    elif param_name == "friction":
                        friction = value
                    elif param_name == "elasticity":
                        elasticity = value
                    elif param_name == "wetness":
                        wetness = value
                    elif param_name == "tear_intensity":
                        tear = value
                    elif param_name == "tension":
                        tension = value

        # 转换为皮肤工厂的专属物理量
        skin_wetness = max(0.0, min(1.0, wetness * self.config["wetness_gain"]))
        skin_smoothness = max(0.0, min(1.0, (1.0 - friction) * self.config["smoothness_gain"]))
        skin_warmth = max(0.0, min(1.0, elasticity * self.config["warmth_gain"]))

        return {"wetness": skin_wetness, "smoothness": skin_smoothness, "warmth": skin_warmth}

    def _generate_base_normal(self, wetness, smoothness, batch, H, W, device, external_normal_kernel):
        if external_normal_kernel and getattr(external_normal_kernel, "kernel", None) is not None:
            return external_normal_kernel.infer(torch.ones((batch, 2, H, W), device=device) * wetness, target_channels=3)
        # 基础法线底盘：不再使用固定的 0.5 中性值，而是基于物理参数生成
        base_z = 0.5 + wetness * 0.5
        base = torch.tensor([0.5, 0.5, base_z], device=device).view(1, 3, 1, 1).repeat(batch, 1, H, W)
        if smoothness < 0.8:
            base = torch.clamp(base + torch.randn((batch, 3, H, W), device=device) * (1.0 - smoothness) * 0.05, 0.0, 1.0)
        return base

    def _generate_extra_channels(self, wetness, smoothness, warmth, batch, num_channels, H, W, device):
        weights = []
        for i in range(num_channels):
            if i % 3 == 0:
                weights.append(wetness)
            elif i % 3 == 1:
                weights.append(smoothness)
            else:
                weights.append(warmth)
        base = torch.tensor(weights, device=device).view(1, num_channels, 1, 1).repeat(batch, 1, H, W)
        noise = torch.randn((batch, num_channels, H, W), device=device) * 0.01
        return torch.clamp(base + noise, 0.0, 1.0)

    def _align_to_target(self, tensor, target_shape):
        B, C, H, W = target_shape
        if tensor.shape[-2:] != (H, W):
            tensor = F.interpolate(tensor, size=(H, W), mode="bilinear", align_corners=False)
        return tensor if tensor.shape[0] == B else (tensor.repeat(B, 1, 1, 1) if tensor.shape[0] == 1 else tensor[:B])

    def _get_default_component(self, batch, channels, H, W, device):
        return {"component_id": "fallback", "tensor": torch.zeros((batch, channels, H, W), device=device)}