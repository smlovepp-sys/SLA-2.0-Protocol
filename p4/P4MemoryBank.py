# P4MemoryBank.py (显存零漏/长时序平滑自适应完美版 + 实时性优化)
import torch
import gc
from typing import Optional, Dict, List

print(">>> P4MemoryBank 工业级轻量稳压版已全面激活 <<<")

class P4MemoryBank:
    def __init__(self, base_beta: float = 0.3, energy_sensitivity: float = 0.85):
        self.base_beta = base_beta
        self.energy_sensitivity = energy_sensitivity
        
        self._prev_rgb: Optional[torch.Tensor] = None
        self._smoothed_rgb: Optional[torch.Tensor] = None
        
        self.prev_latent_stats: Dict[str, Optional[torch.Tensor]] = {
            "mean": None,
            "std": None
        }
        self._current_energy = 0.0

        # 动态基准演进控制
        self.target_std: Optional[float] = None
        self.init_std: Optional[float] = None
        self.smooth_scale: float = 1.0
        self.std_history: List[float] = []
        self.max_history: int = 12  # 略微拉长历史窗口提高长时序稳定性
        self.prev_suppression: Optional[torch.Tensor] = None

        # 存储标量特征而非全量大张量，彻底封死显存泄漏
        self.impact_scalar_history: List[float] = []
        self.max_impact_history: int = 8

        self._progress: Optional[float] = None

    def set_progress(self, progress: float):
        self._progress = progress

    def get_progress(self) -> Optional[float]:
        return self._progress

    def get_init_std(self) -> Optional[float]:
        return self.init_std

    def lock_target_std(self, latent: torch.Tensor):
        current_frame_std = latent.std().item()
        if self.init_std is None:
            self.init_std = current_frame_std
            
        # 极其平滑的滚动演进策略（EMA 锚定）
        if self.target_std is None:
            self.target_std = current_frame_std
        else:
            # 允许目标基准以 5% 的微弱速率随新场景演进，既防闪烁，又能自适应长视频镜头切换
            self.target_std = self.target_std * 0.95 + current_frame_std * 0.05

    def _update_std_history(self, current_std: float):
        self.std_history.append(current_std)
        if len(self.std_history) > self.max_history:
            self.std_history.pop(0)

    def _get_dynamic_threshold(self) -> float:
        if len(self.std_history) < 4:
            return 1.2
        arr = torch.tensor(self.std_history, dtype=torch.float32)
        mean_val = arr.mean().item()
        std_val = arr.std().item()
        cv = std_val / (mean_val + 1e-6)
        cv_clamped = max(0.0, min(cv * 1.5, 0.45))
        return 1.05 + cv_clamped

    def _get_dynamic_momentum(self, current_std: float) -> float:
        if self.target_std is None or self.target_std <= 0:
            return 0.9
        deviation = abs(current_std / self.target_std - 1.0)
        clamped = max(0.0, min(deviation * 0.5, 0.25))
        return 0.95 - clamped

    def get_scale_factor(self, current_std: float) -> float:
        if self.target_std is None or self.target_std <= 0:
            return 1.0
        self._update_std_history(current_std)
        threshold = self._get_dynamic_threshold()
        momentum = self._get_dynamic_momentum(current_std)
        
        if current_std > self.target_std * threshold:
            raw_scale = self.target_std / (current_std + 1e-6)
            self.smooth_scale = momentum * self.smooth_scale + (1.0 - momentum) * raw_scale
            return self.smooth_scale
        else:
            self.smooth_scale = momentum * self.smooth_scale + (1.0 - momentum) * 1.0
            return self.smooth_scale

    def update_impact_variance(self, impact_geo: torch.Tensor, impact_mat: torch.Tensor) -> float:
        with torch.no_grad():
            geo_energy = impact_geo.detach().pow(2).mean().sqrt().item()
            mat_energy = impact_mat.detach().pow(2).mean().sqrt().item()
            combined_energy = (geo_energy + mat_energy) * 0.5
            
        self.impact_scalar_history.append(combined_energy)
        if len(self.impact_scalar_history) > self.max_impact_history:
            self.impact_scalar_history.pop(0)
            
        if len(self.impact_scalar_history) < 4:
            return 0.1
            
        arr = torch.tensor(self.impact_scalar_history, dtype=torch.float32)
        return arr.std().item()

    def smooth_suppression(self, suppression: torch.Tensor, momentum: float = 0.9) -> torch.Tensor:
        if self.prev_suppression is None or self.prev_suppression.shape != suppression.shape:
            self.prev_suppression = suppression.detach().clone()
        else:
            if self.prev_suppression.device != suppression.device:
                self.prev_suppression = self.prev_suppression.to(suppression.device)
            self.prev_suppression = momentum * self.prev_suppression + (1.0 - momentum) * suppression
        return self.prev_suppression

    def update(self, current_rgb: torch.Tensor, current_latent: Optional[torch.Tensor] = None,
               energy_map: Optional[torch.Tensor] = None):
        if current_rgb is None:
            return None
            
        if energy_map is not None:
            self.set_physical_energy(energy_map)
        if current_latent is not None:
            self.lock_target_std(current_latent)
            
        if self._smoothed_rgb is not None and self._smoothed_rgb.device != current_rgb.device:
            self._smoothed_rgb = self._smoothed_rgb.to(current_rgb.device)

        self._prev_rgb = current_rgb.detach().clone()
        beta = self.dynamic_beta
        
        if self._smoothed_rgb is None or self._smoothed_rgb.shape != current_rgb.shape:
            self._smoothed_rgb = current_rgb.detach().clone()
        else:
            self._smoothed_rgb = self._smoothed_rgb * beta + current_rgb * (1.0 - beta)
            
        if current_latent is not None:
            with torch.no_grad():
                self.prev_latent_stats["mean"] = current_latent.mean(dim=(-2, -1), keepdim=True).detach()
                self.prev_latent_stats["std"] = current_latent.std(dim=(-2, -1), keepdim=True).detach()
                
        return self._prev_rgb

    def get_last(self, full_precision: bool = True) -> Optional[torch.Tensor]:
        return self._prev_rgb

    def get_smoothed(self, full_precision: bool = True) -> Optional[torch.Tensor]:
        if self._smoothed_rgb is None:
            return None
        return self._smoothed_rgb.clone()

    def get_latent_stats(self, target_device: Optional[torch.device] = None) -> Dict[str, Optional[torch.Tensor]]:
        if target_device is not None:
            out_stats = {}
            for k, v in self.prev_latent_stats.items():
                out_stats[k] = v.to(target_device) if v is not None else None
            return out_stats
        return self.prev_latent_stats

    def get_current_energy(self) -> float:
        return self._current_energy

    def set_physical_energy(self, energy_map: torch.Tensor):
        with torch.no_grad():
            energy_val = energy_map.float().mean().item()
            self._current_energy = self._current_energy * 0.8 + energy_val * 0.2
            self._current_energy = max(0.0, min(1.0, self._current_energy))

    @property
    def dynamic_beta(self) -> float:
        energy = max(0.0, min(1.0, self._current_energy))
        dynamic_beta = self.base_beta * (1.0 - energy * self.energy_sensitivity)
        return max(0.05, min(0.95, dynamic_beta))

    def clear_temporal_memory(self):
        self._prev_rgb = None
        self._smoothed_rgb = None
        for k in self.prev_latent_stats:
            self.prev_latent_stats[k] = None

    def flush(self):
        """确保完全清空历史缓冲区，防止脏数据带入新帧"""
        self._prev_rgb = None
        self._smoothed_rgb = None
        self.prev_latent_stats = {"mean": None, "std": None}
        self._current_energy = 0.0
        self.target_std = None
        self.init_std = None
        self.smooth_scale = 1.0
        self.std_history.clear()
        self.impact_scalar_history.clear()
        self.prev_suppression = None
        self._progress = None
        # 强制显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def reset(self):
        self.flush()