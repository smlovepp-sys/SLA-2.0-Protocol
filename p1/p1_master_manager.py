# p1_master_manager.py（调度装配专用版：信息密度加权 + 诊断优化，移除二次缩小，精简打印）
import os, sys, math, torch, logging
from typing import Dict, List, Tuple, Any, Optional

from .p1_skin_factory import SkinFactory
from .p1_joint_factory import JointFactory
from .p1_fabric_factory import FabricFactory
from .p1_fluid_factory import FluidFactory

logger = logging.getLogger(__name__)

PARAM_BOUNDS = {
    "stiffness": (0.0, 1.0),
    "friction": (0.0, 1.0),
    "elasticity": (0.0, 1.0),
    "wetness": (0.0, 1.0),
    "tear_intensity": (0.0, 1.0),
    "tension": (-1.0, 1.0),
    "body_type": (0.0, 1.0),
}

DEFAULT_PHYSICS_CONFIG = {
    "energy_ceiling": 1.2,
    "conflict_pairs": [("stiffness", "elasticity")],
    "strategy": "proportional_scaling",
    "dynamic_alpha_base": 0.3,
    "dynamic_alpha_confidence_gain": 0.2,
    "evolution_rate_wetness_gain": 0.5,
    "evolution_rate_delta_gain": 2.0,
    "min_factory_channels": 2,
    "hysteresis_buffer_ratio": 0.05,
    "hysteresis_persistence_frames": 3,
    "nan_consecutive_threshold": 5,
}

NEUTRAL_PARAMS = {
    "stiffness": 0.5,
    "friction": 0.5,
    "elasticity": 0.5,
    "wetness": 0.0,
    "tear_intensity": 0.0,
    "tension": 0.0,
    "body_type": 0.5,
}


class P1MasterManager:
    def __init__(self,
                 raw_config: Optional[Dict[str, Any]] = None,
                 global_resolution: Tuple[int, int] = (64, 64),
                 target_model_channels: int = 48,
                 target_device: str = "cpu"):

        self.raw_config = raw_config if raw_config is not None else {}
        self.global_resolution = global_resolution          # 全尺寸 LAT 分辨率
        self.target_model_channels = target_model_channels
        self.target_device = torch.device(target_device)

        self.physics_config = DEFAULT_PHYSICS_CONFIG.copy()
        user_physics = self.raw_config.get("physics_config", {})
        self.physics_config.update(user_physics)

        self.bridge_sync = NeuralBridge()
        self.qc_inspector = QCInspector()
        self.final_assembly = FinalAssemblyWorkshop()
        self.customs = ChannelCustoms()

        self.normal_kernel = getattr(sys.modules[__name__], "ExternalNormalKernel", None)

        self.skin_factory = SkinFactory()
        self.joint_factory = JointFactory()
        self.fabric_factory = FabricFactory()
        self.fluid_factory = FluidFactory()

        self.last_valid_tensor: Optional[torch.Tensor] = None
        self.frame_counter: int = 0
        self.previous_wetness: float = 0.0

        self.factory_quality_ema: Dict[str, float] = {
            "skin_factory": 0.5,
            "joint_factory": 0.5,
            "fabric_factory": 0.5,
            "fluid_factory": 0.5,
        }

        self._last_valid_components: Dict[str, torch.Tensor] = {}
        self.nan_fallback_count: int = 0
        self._nan_streak: Dict[str, int] = {}
        self._force_neutral_factories: set = set()

        print(f"[P1Master] 初始化完成。通道: {self.target_model_channels}, 设备: {self.target_device}")

    def _health_report(self):
        if self.nan_fallback_count > 0:
            print(f"[P1健康] NaN修复: {self.nan_fallback_count} | 连续NaN: {self._nan_streak} | 中性工厂: {self._force_neutral_factories}")
        if self.nan_fallback_count > 50:
            print("⚠️ 高频 NaN，强制清理保护状态。")
            self.nan_fallback_count = 0
            self._nan_streak.clear()
            self._force_neutral_factories.clear()

    def _update_quality_ema(self, quality_scores: Dict[str, float]):
        alpha_q = 0.2
        for fac, score in quality_scores.items():
            if fac in self.factory_quality_ema:
                self.factory_quality_ema[fac] = alpha_q * score + (1 - alpha_q) * self.factory_quality_ema[fac]
            else:
                self.factory_quality_ema[fac] = score

    def _dynamic_channel_budget(self, components: List[Dict], injected_tokens: Dict[str, List]) -> Dict[str, int]:
        total_ch = self.target_model_channels
        min_ch = self.physics_config.get("min_factory_channels", 2)
        factory_names = [c.get("factory") for c in components if c.get("factory") in injected_tokens]
        wetness = self.previous_wetness

        weights = {}
        for fac in factory_names:
            q = self.factory_quality_ema.get(fac, 0.5)
            w = max(q, 0.1)
            if "fluid" in fac:
                w = max(w, 0.25) + wetness * 0.3
            elif "fabric" in fac and wetness > 0.5:
                w += 0.2
            param_count = len(injected_tokens.get(fac, []))
            w *= math.sqrt(max(param_count, 1))
            weights[fac] = w

        total_w = sum(weights.values()) or 1
        budget = {}
        allocated = 0
        for fac in factory_names:
            share = max(min_ch, int(weights[fac] / total_w * total_ch))
            budget[fac] = share
            allocated += share

        diff = total_ch - allocated
        if diff > 0:
            sorted_facs = sorted(factory_names, key=lambda f: weights[f], reverse=True)
            for fac in sorted_facs:
                if diff == 0: break
                budget[fac] += 1
                diff -= 1
        elif diff < 0:
            over = -diff
            sorted_facs = sorted(factory_names, key=lambda f: weights[f])
            for fac in sorted_facs:
                while over > 0 and budget[fac] > min_ch:
                    budget[fac] -= 1
                    over -= 1
                if over == 0: break
        return budget

    def process_frame(self,
                      conditioning_tensor: torch.Tensor,
                      hidden_states: torch.Tensor,
                      global_physics: Dict[str, float] = None,
                      factory_token_map: Dict[str, List[Tuple[str, Dict[str, float]]]] = None,
                      external_timestamp: Optional[float] = None) -> Dict[str, Any]:
        self.frame_counter += 1
        self._health_report()

        if global_physics is None:
            global_physics = dict(NEUTRAL_PARAMS)
        if factory_token_map is None:
            factory_token_map = {}
        self.previous_wetness = global_physics.get("wetness", 0.0)

        # 令牌格式转换（兼容内部工厂接口）
        injected_tokens = {"skin_factory": [], "joint_factory": [], "fabric_factory": [], "fluid_factory": []}
        for fac, tokens in factory_token_map.items():
            if fac not in injected_tokens:
                injected_tokens[fac] = []
            for token in tokens:
                if isinstance(token, (list, tuple)) and len(token) == 2 and isinstance(token[1], dict):
                    obj_name, params = token
                    for pname, pval in params.items():
                        injected_tokens[fac].append((f"{obj_name}_{pname}", float(pval)))
                else:
                    injected_tokens[fac].append(token)

        components = []
        def safe_produce(factory_name, produce_fn, *args, **kwargs):
            if factory_name in self._force_neutral_factories:
                kwargs["global_physics"] = dict(NEUTRAL_PARAMS)
            try:
                comp = produce_fn(*args, **kwargs)
                tensor = comp.get("tensor")
                if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                    logger.warning(f"工厂 {factory_name} 输出包含 NaN/Inf，尝试替换")
                    self._nan_streak[factory_name] = self._nan_streak.get(factory_name, 0) + 1
                    threshold = self.physics_config.get("nan_consecutive_threshold", 5)
                    if self._nan_streak[factory_name] >= threshold:
                        logger.error(f"工厂 {factory_name} 连续 {threshold} 帧 NaN，触发强制重置")
                        self.factory_quality_ema[factory_name] = 0.0
                        self._force_neutral_factories.add(factory_name)
                        self._nan_streak[factory_name] = 0
                    if factory_name in self._last_valid_components:
                        comp["tensor"] = self._last_valid_components[factory_name].clone()
                    else:
                        comp["tensor"] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
                    self.nan_fallback_count += 1
                else:
                    if factory_name in self._nan_streak:
                        del self._nan_streak[factory_name]
                    if factory_name in self._force_neutral_factories:
                        self._force_neutral_factories.remove(factory_name)
                        self.factory_quality_ema[factory_name] = 0.3
                    if tensor is not None:
                        self._last_valid_components[factory_name] = tensor.clone()
                comp["tokens"] = injected_tokens.get(factory_name, [])
                return comp
            except Exception as e:
                logger.error(f"工厂 {factory_name} 生产失败: {e}")
                if factory_name in self._last_valid_components:
                    return {"factory": factory_name, "tensor": self._last_valid_components[factory_name].clone(), "tokens": []}
                return None

        # 各工厂生产
        skin_comp = safe_produce("skin_factory", self.skin_factory.produce,
                                 factory_tokens=injected_tokens.get("skin_factory", []),
                                 global_physics=global_physics,
                                 global_resolution=self.global_resolution,
                                 target_channels=16, target_batch=1,
                                 device=self.target_device,
                                 external_normal_kernel=self.normal_kernel,
                                 time_factor=external_timestamp or 0.0)
        if skin_comp: components.append(skin_comp)

        joint_comp = safe_produce("joint_factory", self.joint_factory.produce,
                                  factory_tokens=injected_tokens.get("joint_factory", []),
                                  global_physics=global_physics,
                                  normal_base_shape=(1, 3, *self.global_resolution),
                                  delta_shape=(1, 2, *self.global_resolution),
                                  device=self.target_device,
                                  external_normal_kernel=self.normal_kernel,
                                  time_factor=external_timestamp or 0.0)
        if joint_comp: components.append(joint_comp)

        fabric_comp = safe_produce("fabric_factory", self.fabric_factory.produce,
                                   factory_tokens=injected_tokens.get("fabric_factory", []),
                                   global_physics=global_physics,
                                   global_resolution=self.global_resolution,
                                   target_channels=8, target_batch=1,
                                   device=self.target_device,
                                   external_normal_kernel=self.normal_kernel,
                                   time_factor=external_timestamp or 0.0)
        if fabric_comp: components.append(fabric_comp)

        fluid_comp = safe_produce("fluid_factory", self.fluid_factory.produce,
                                  factory_tokens=injected_tokens.get("fluid_factory", []),
                                  global_physics=global_physics,
                                  global_resolution=self.global_resolution,
                                  target_channels=12, target_batch=1,
                                  device=self.target_device,
                                  external_normal_kernel=self.normal_kernel,
                                  time_factor=external_timestamp or 0.0)
        if fluid_comp: components.append(fluid_comp)

        # 质检
        sync_package = self.bridge_sync.sync(components, expected_factories=["skin_factory", "joint_factory", "fabric_factory", "fluid_factory"])
        validated_comps = []
        for comp in sync_package.get("components", []):
            vc = self.qc_inspector.inspect(comp)
            if vc and vc.get("tensor") is not None and vc["tensor"].abs().max() > 0:
                validated_comps.append(vc)

        temp_quality = {}
        for comp in validated_comps:
            t = comp["tensor"]
            fac = comp["factory"]
            if t is not None and t.numel() > 0:
                temp_quality[fac] = min(1.0, t.float().std(dim=(-2, -1)).mean().item() / 0.1)
        self._update_quality_ema(temp_quality)
        validated_comps.sort(key=lambda x: self.factory_quality_ema.get(x["factory"], 0.5), reverse=True)

        budget = self._dynamic_channel_budget(validated_comps, injected_tokens)

        assembled = self.final_assembly.assemble(validated_comps, self.target_model_channels, budget, global_physics)
        final_tensor = assembled.get("final_tensor")

        if final_tensor is not None and (torch.isnan(final_tensor).any() or torch.isinf(final_tensor).any()):
            final_tensor = self.last_valid_tensor.clone() if self.last_valid_tensor is not None else torch.zeros_like(final_tensor)
        elif final_tensor is not None:
            self.last_valid_tensor = final_tensor.clone()

        if self.customs is not None:
            command = {"target_channels": self.target_model_channels, "target_dtype": "float32",
                       "target_device": str(self.target_device), "channel_adapter_strategy": "zero"}
            clear_packet = self.customs.clearance(command, assembled)
            final_tensor_cleared = clear_packet.get("p1_final_tensor", final_tensor)
        else:
            final_tensor_cleared = final_tensor

        final_output = final_tensor_cleared

        cipher_map = {"stiffness": "Sf", "friction": "Sm", "elasticity": "L",
                      "wetness": "Lh", "tear_intensity": "Fo", "tension": "Ft", "body_type": "Bd"}
        global_ciphers = {cipher_map.get(k, k): v for k, v in global_physics.items()}

        return {
            "p1_stat": "success",
            "p1_shadow_frames": final_output,
            "assets": {
                "factory_tokens": injected_tokens,
                "global_physics": global_physics,
                "global_ciphers": global_ciphers
            },
            "ciphers": list(global_ciphers.keys())
        }


class NeuralBridge:
    def sync(self, components, expected_factories):
        components.sort(key=lambda x: x.get("layer_depth", 0))
        return {"components": components}

class QCInspector:
    def inspect(self, component):
        tensor = component.get("tensor")
        if tensor is not None and isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                component["tensor"] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
            return component
        return None

class FinalAssemblyWorkshop:
    def assemble(self, components, target_channels=48, channel_budget=None, global_params=None):
        if not components:
            return {"final_tensor": None}
        ref_tensor = components[0]["tensor"]
        B, _, H, W = ref_tensor.shape
        device = ref_tensor.device
        master = torch.zeros((B, target_channels, H, W), device=device)

        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        spatial_basis = (0.5 + 0.5 * torch.sin(x * 1.5) * torch.cos(y * 1.5)).unsqueeze(0).unsqueeze(0)
        roughness = global_params.get("friction", 0.5) if global_params else 0.5

        if channel_budget:
            total_alloc = sum(channel_budget.values())
            if total_alloc != target_channels:
                factor = target_channels / total_alloc
                channel_budget = {k: max(1, int(v * factor)) for k, v in channel_budget.items()}
                diff = target_channels - sum(channel_budget.values())
                if diff != 0:
                    fac_with_max = max(channel_budget, key=lambda f: channel_budget[f])
                    channel_budget[fac_with_max] += diff

        cur = 0
        for comp in components:
            t = comp["tensor"]
            t_normalized = torch.tanh((t - t.mean()) / (t.std() + 1e-6))
            modulated_t = t_normalized + (spatial_basis - 0.5) * 0.05 * roughness
            c_len = min(t.shape[1], channel_budget.get(comp["factory"], t.shape[1]))
            if cur + c_len <= target_channels:
                master[:, cur:cur+c_len] = modulated_t[:, :c_len]
                cur += c_len
        master = (master - master.mean()) / (master.std() + 1e-6)
        return {"final_tensor": master, "components_count": len(components)}

class ChannelCustoms:
    def clearance(self, command, assembled):
        final_tensor = assembled.get("final_tensor")
        if final_tensor is not None:
            target_device = torch.device(command.get("target_device", "cpu"))
            if final_tensor.device != target_device:
                final_tensor = final_tensor.to(target_device)
        return {"p1_final_tensor": final_tensor}