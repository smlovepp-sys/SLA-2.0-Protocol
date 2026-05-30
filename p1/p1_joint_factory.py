# p1_joint_factory.py
# P1 关节零件厂：输出法线修正增量 + 位移场 + 影响力遮罩
# V2.4 统一接口版：produce(**kwargs)，消除签名特异性

import hashlib
import threading
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F


class JointFactory:
    """
    关节零件厂：生产形变插件与位移场场强增量。
    适配统一 **kwargs 接口，无物理激活时输出全零张量。
    """

    def __init__(self):
        self.factory_name = "joint_factory"
        self._cache = {}
        self._cache_lock = threading.RLock()

    # ---------- 内部辅助计算 ----------
    def _compute_motion_intensity(self, global_physics: Dict[str, float], factory_tokens: List) -> float:
        """
        根据全局物理参数和工厂令牌计算运动强度。
        令牌可以覆盖全局参数（兼容新旧格式）。
        """
        stiffness = global_physics.get("stiffness", 0.5)
        tension = global_physics.get("tension", 0.0)
        friction = global_physics.get("friction", 0.5)

        # 处理令牌覆盖（兼容 (name, dict) 和 (name, value) 格式）
        for token in factory_tokens:
            if isinstance(token, (list, tuple)) and len(token) == 2:
                if isinstance(token[1], dict):
                    params = token[1]
                    if "stiffness" in params:
                        stiffness = float(params["stiffness"])
                    if "tension" in params:
                        tension = float(params["tension"])
                    if "friction" in params:
                        friction = float(params["friction"])
                else:
                    param_name = token[0]
                    value = token[1]
                    if "_" in param_name:
                        param_name = param_name.split("_")[-1]
                    if param_name == "stiffness":
                        stiffness = value
                    elif param_name == "tension":
                        tension = value
                    elif param_name == "friction":
                        friction = value

        rigidity_factor = 1.0 - stiffness
        tension_factor = abs(tension)
        intensity = 0.1 + rigidity_factor * 0.5 + tension_factor * 0.4
        return max(0.0, min(1.0, intensity))

    def _extract_joint_parts(self, factory_tokens: List) -> List[str]:
        keywords = ["elbow", "knee", "wrist", "ankle", "shoulder", "hip", "neck", "spine", "jaw", "finger", "attach"]
        parts = []
        for token in factory_tokens:
            if isinstance(token, (list, tuple)) and len(token) >= 1:
                obj_name = token[0] if isinstance(token[0], str) else ""
            else:
                obj_name = str(token)
            t = obj_name.lower()
            for kw in keywords:
                if kw in t and kw not in parts:
                    parts.append(kw)
        if not parts:
            parts = ["attach"]
        return parts

    def _generate_rig_and_delta(self,
                                batch: int,
                                H: int,
                                W: int,
                                intensity: float,
                                device: torch.device,
                                time_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if intensity <= 0.01:
            return (torch.zeros(batch, 2, H, W, device=device),
                    torch.zeros(batch, 1, H, W, device=device))

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        xx = xx.unsqueeze(0).repeat(batch, 1, 1).unsqueeze(1)
        yy = yy.unsqueeze(0).repeat(batch, 1, 1).unsqueeze(1)

        center_x = 0.0 + 0.1 * torch.sin(torch.tensor(time_factor, device=device))
        center_y = 0.0 + 0.1 * torch.cos(torch.tensor(time_factor, device=device))

        dist = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        sigma = 0.4 + intensity * 0.2
        rig_map = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
        rig_map = torch.clamp(rig_map, 0.0, 1.0)

        delta_x = - (yy - center_y) * rig_map * intensity * 0.2
        delta_y = (xx - center_x) * rig_map * intensity * 0.2
        joint_delta = torch.cat([delta_x, delta_y], dim=1)

        return joint_delta, rig_map

    def _delta_to_normal_delta(self,
                               joint_delta: torch.Tensor,
                               normal_base_shape: Tuple[int, int, int, int],
                               device: torch.device,
                               external_normal_kernel: Optional[Any] = None) -> torch.Tensor:
        B, C, H, W = normal_base_shape

        if external_normal_kernel and getattr(external_normal_kernel, "kernel", None) is not None:
            ext_out = external_normal_kernel.infer(joint_delta, target_channels=3)
            if ext_out is not None:
                return ext_out

        grad_x = torch.abs(joint_delta[:, 0:1, :, :-1] - joint_delta[:, 0:1, :, 1:])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")

        grad_y = torch.abs(joint_delta[:, 1:2, :-1, :] - joint_delta[:, 1:2, 1:, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")

        normal_delta_x = torch.clamp(grad_x * 2.0, 0.0, 0.5)
        normal_delta_y = torch.clamp(grad_y * 2.0, 0.0, 0.5)
        normal_delta_z = torch.clamp(1.0 - (grad_x + grad_y) * 0.5, 0.3, 1.0)

        return torch.cat([normal_delta_x, normal_delta_y, normal_delta_z], dim=1)

    def _expand_normal_delta(self, normal_delta: torch.Tensor, remaining_channels: int) -> torch.Tensor:
        B, _, H, W = normal_delta.shape
        repeats = (remaining_channels + 2) // 3
        extended = normal_delta.repeat(1, repeats, 1, 1)
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
        接收所有参数通过关键字传递，主要提取：
        - factory_tokens: List
        - global_physics: Dict[str, float]
        - normal_base_shape: Tuple[int,int,int,int]  (默认 (1,3,64,64))
        - delta_shape: Tuple[int,int,int,int]        (默认 (1,2,64,64))
        - device: torch.device                        (默认 cpu)
        - external_normal_kernel: Optional
        - time_factor: float                          (默认 0.0)
        """
        factory_tokens = kwargs.get("factory_tokens", [])
        global_physics = kwargs.get("global_physics", {})
        normal_base_shape = kwargs.get("normal_base_shape", (1, 3, 64, 64))
        delta_shape = kwargs.get("delta_shape", (1, 2, 64, 64))
        device = kwargs.get("device", torch.device("cpu"))
        external_normal_kernel = kwargs.get("external_normal_kernel", None)
        time_factor = kwargs.get("time_factor", 0.0)

        target_batch = normal_base_shape[0]
        expected_total_channels = 8  # 关节厂固定输出 8 通道
        H, W = normal_base_shape[-2], normal_base_shape[-1]

        # 计算运动强度（基于 global_physics 和 tokens）
        motion_intensity = self._compute_motion_intensity(global_physics, factory_tokens)

        # 无令牌时输出零张量
        if len(factory_tokens) == 0:
            full_tensor = torch.zeros(target_batch, expected_total_channels, H, W, device=device)
            return {
                "component_id": hashlib.md5(f"{self.factory_name}_neutral_{target_batch}_{H}_{W}".encode()).hexdigest()[:12],
                "tensor": full_tensor,
                "joint_delta": torch.zeros(target_batch, 2, H, W, device=device),
                "rig_map": torch.zeros(target_batch, 1, H, W, device=device),
                "normal_delta": torch.zeros(target_batch, 3, H, W, device=device),
                "factory": self.factory_name,
                "layer_depth": 1
            }

        joint_parts = self._extract_joint_parts(factory_tokens)

        token_hash = hashlib.md5(str(factory_tokens).encode()).hexdigest()[:8]
        cache_key = f"{target_batch}_{H}_{W}_{motion_intensity:.3f}_{token_hash}_{device}_{time_factor:.2f}"

        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        joint_delta, rig_map = self._generate_rig_and_delta(
            target_batch, H, W, motion_intensity, device, time_factor
        )

        normal_delta = self._delta_to_normal_delta(
            joint_delta, normal_base_shape, device, external_normal_kernel=external_normal_kernel
        )

        front_package = torch.cat([joint_delta, rig_map], dim=1)

        if expected_total_channels > 3:
            remaining = expected_total_channels - 3
            extended_normals = self._expand_normal_delta(normal_delta, remaining)
            full_tensor = torch.cat([front_package, extended_normals], dim=1)
        else:
            full_tensor = front_package[:, :expected_total_channels, :, :]

        final_shape = (target_batch, expected_total_channels, H, W)
        full_tensor = self._align_to_target(full_tensor, final_shape)

        instance_id = hashlib.md5(
            f"{self.factory_name}_{motion_intensity:.3f}_{target_batch}_{H}_{W}".encode()
        ).hexdigest()[:12]

        component = {
            "component_id": instance_id,
            "tensor": full_tensor,
            "joint_delta": joint_delta,
            "rig_map": rig_map,
            "normal_delta": normal_delta,
            "factory": self.factory_name,
            "layer_depth": 1
        }

        with self._cache_lock:
            self._cache[cache_key] = component

        return component