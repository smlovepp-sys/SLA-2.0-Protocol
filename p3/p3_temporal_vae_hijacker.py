# p3_temporal_vae_hijacker.py (物理场直通版 + 惯性保持 + 缺失诊断)
import torch
import torch.nn.functional as F
import math
import os
import tempfile
import threading
import concurrent.futures
import json
import shutil
from safetensors.torch import save_file, load_file
from typing import Dict, Any, Optional, Tuple, List

class P3TemporalVAEHijacker:
    def __init__(self, fps: float = 30.0, alpha: float = 0.15, vae_mode: str = "auto",
                 vae_model: Optional[torch.nn.Module] = None, cache_dir: Optional[str] = None,
                 max_workers: int = 4, base_influence_scale: float = 1.0):
        self.fps = fps
        self.alpha = alpha
        self.vae_mode = vae_mode
        self.base_influence_scale = base_influence_scale
        self.dt_ref = 1.0 / self.fps
        self.full_channels = 48

        if vae_model is not None and hasattr(vae_model, 'in_channels'):
            self.target_channels = vae_model.in_channels
        else:
            self.target_channels = self._resolve_target_channels(vae_mode)

        print(f"[Hijacker] 48ch 物理场直通切片架构 | 目标VAE通道: {self.target_channels}")

        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="p3_hijacker_")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.manifest: Dict[str, Any] = {
            "total_frames": 0,
            "vae_mode": self.vae_mode,
            "fps": self.fps
        }
        self.disk_paths: Dict[int, Tuple[str, str]] = {}
        self._memory_buffer: Optional[Dict[str, Any]] = None
        self._memory_buffer_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # 新增：物理状态记忆，用于缺失字段的惯性保持
        self._last_physics_state = {"stiffness": 0.5, "wetness": 0.0, "tear_intensity": 0.0}

    def _resolve_target_channels(self, vae_mode: str) -> int:
        if vae_mode == "4ch": return 4
        elif vae_mode == "16ch": return 16
        elif vae_mode == "48ch": return 48
        else: return 4

    def _write_payload_to_disk(self, frame_idx: int, payload: Dict[str, Any]) -> Tuple[int, str, str]:
        file_patch = os.path.join(self.cache_dir, f"frame_{frame_idx:06d}_tri.safetensors")
        file_meta = os.path.join(self.cache_dir, f"frame_{frame_idx:06d}_meta.json")
        try:
            save_file({"block_A": payload["block_A"], "block_B": payload["block_B"], "block_C": payload["block_C"]}, file_patch)
            with open(file_meta, 'w') as f:
                json.dump(payload["evolved_params"], f)
        except Exception as e:
            print(f"[Hijacker] 写入帧 {frame_idx} 失败: {e}")
            raise
        return frame_idx, file_patch, file_meta

    def precompute_sequence_from_states(self,
                                        total_frames: int,
                                        ciphers: List[str],
                                        base_params: Dict[str, float],
                                        all_states: List[Dict[str, Any]],
                                        spatial_size: Tuple[int, int],
                                        progress_callback=None,
                                        strength_scale: float = 1.0,
                                        seed: int = 42,
                                        precomputed_physics: Optional[torch.Tensor] = None) -> None:
        H, W = spatial_size
        self.manifest.update({
            "total_frames": total_frames,
            "spatial_size": [H, W],
            "ciphers": ciphers,
            "base_params": base_params,
            "vae_mode": self.vae_mode,
            "target_channels": self.target_channels,
            "fps": self.fps,
        })
        self.disk_paths.clear()
        self._memory_buffer = None
        futures = []

        for i in range(total_frames):
            state_raw = all_states[i] if i < len(all_states) else {}
            physics_params = self._extract_physics_params(state_raw, base_params)

            if precomputed_physics is not None and i < precomputed_physics.shape[0]:
                phys_frame = precomputed_physics[i]  # [48, H, W]
                if phys_frame.shape[-2:] != (H, W):
                    phys_frame = F.interpolate(phys_frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                block_A = phys_frame[0:16].unsqueeze(0).cpu()
                block_B = phys_frame[16:32].unsqueeze(0).cpu()
                block_C = phys_frame[32:48].unsqueeze(0).cpu()
            else:
                # 降级：生成简单非零纹理（仅当 precomputed_physics 缺失时使用）
                block_A = torch.zeros(1, 16, H, W)
                block_B = torch.zeros(1, 16, H, W)
                block_C = torch.zeros(1, 16, H, W)
                t = i / max(total_frames, 1)
                block_A[:, :, :, :] = 0.1 * torch.sin(torch.linspace(0, 2*math.pi, H).view(-1,1) + 
                                                         torch.linspace(0, 2*math.pi, W).view(1,-1) + t * 10)
                block_B[:, :, :, :] = 0.1 * torch.cos(torch.linspace(0, 2*math.pi, H).view(-1,1) * 2 + 
                                                         torch.linspace(0, 2*math.pi, W).view(1,-1) * 3)
                block_C[:, :, :, :] = 0.1 * torch.sin(torch.linspace(0, 2*math.pi, H).view(-1,1) * 0.5 + 
                                                         torch.linspace(0, 2*math.pi, W).view(1,-1) * 2)

            payload = {
                "block_A": block_A,
                "block_B": block_B,
                "block_C": block_C,
                "evolved_params": physics_params
            }
            if i == 0:
                with self._memory_buffer_lock:
                    self._memory_buffer = payload.copy()
            futures.append((i, self._executor.submit(self._write_payload_to_disk, i, payload)))

        for i, future in futures:
            idx, patch_path, meta_path = future.result()
            self.disk_paths[idx] = (patch_path, meta_path)
            if progress_callback:
                progress_callback(idx + 1, total_frames)

    def _extract_physics_params(self, state_raw: Dict[str, Any], base_params: Dict[str, float]) -> Dict[str, float]:
        # 提取当前物理状态
        first_value = next(iter(state_raw.values()), None)
        physics = first_value if isinstance(first_value, dict) else state_raw

        # 检测关键字段是否缺失，如果缺失则打印警告
        if "wetness" not in physics:
            print(f"[Hijacker 警告] 帧物理数据中未找到 'wetness'，保持上一帧记忆。当前 state_raw keys: {list(physics.keys())}")

        # 惯性保持逻辑：优先使用当前帧的值，缺失时继承上一帧，再缺失则使用 base_params
        new_params = {
            "stiffness": physics.get("stiffness", self._last_physics_state.get("stiffness", base_params.get("stiffness", 0.5))),
            "wetness": physics.get("wetness", self._last_physics_state.get("wetness", base_params.get("wetness", 0.0))),
            "tear_intensity": physics.get("tear_intensity", self._last_physics_state.get("tear_intensity", base_params.get("tear_intensity", 0.0)))
        }
        
        # 更新记忆
        self._last_physics_state = new_params
        return new_params

    def request_payload(self, frame_idx: int) -> Dict[str, Any]:
        if frame_idx == 0 and self._memory_buffer is not None:
            with self._memory_buffer_lock:
                return self._memory_buffer.copy()

        if frame_idx not in self.disk_paths:
            spatial = self.manifest.get("spatial_size", [48, 48])
            H, W = spatial[0], spatial[1]
            zero_block = torch.zeros(1, 16, H, W)
            return {
                "block_A": zero_block,
                "block_B": zero_block,
                "block_C": zero_block,
                "evolved_params": {"stiffness": 0.5, "wetness": 0.0, "tear_intensity": 0.0}
            }

        patch_path, meta_path = self.disk_paths[frame_idx]
        try:
            tensors = load_file(patch_path, device="cpu")
            with open(meta_path, 'r') as f:
                evolved = json.load(f)
        except Exception as e:
            print(f"[Hijacker] 读取帧 {frame_idx} 失败: {e}，返回空张量。")
            spatial = self.manifest.get("spatial_size", [48, 48])
            H, W = spatial[0], spatial[1]
            zero_block = torch.zeros(1, 16, H, W)
            return {
                "block_A": zero_block,
                "block_B": zero_block,
                "block_C": zero_block,
                "evolved_params": {"stiffness": 0.5, "wetness": 0.0, "tear_intensity": 0.0}
            }
        return {
            "block_A": tensors["block_A"],
            "block_B": tensors["block_B"],
            "block_C": tensors["block_C"],
            "evolved_params": evolved
        }

    def apply_temporal_hijack_with_step(self, latent: torch.Tensor, injection_payload: Dict[str, Any],
                                        current_step: int, total_steps: int, dt: float = None) -> torch.Tensor:
        with torch.no_grad():
            progress = current_step / total_steps
            if progress > 0.6:
                return latent

            block_A = injection_payload["block_A"]
            block_B = injection_payload["block_B"]
            B, C_latent, H, W = latent.shape

            step_fade = 1.0 if progress < 0.3 else (1.0 - (progress - 0.3) / 0.3)
            if dt is None:
                dt = self.dt_ref
            dynamic_scale = self._get_dynamic_scale(dt) * step_fade

            dev_A = block_A.to(latent.device, dtype=latent.dtype)
            dev_B = block_B.to(latent.device, dtype=latent.dtype)

            mixed = torch.zeros(B, C_latent, dev_A.shape[2], dev_A.shape[3],
                                device=latent.device, dtype=latent.dtype)
            for c in range(C_latent):
                if c % 2 == 0:
                    mixed[:, c, :, :] = dev_A[:, c // 2, :, :]
                else:
                    mixed[:, c, :, :] = dev_B[:, c // 2, :, :]

            if mixed.shape[2:] != (H, W):
                mixed = F.interpolate(mixed, size=(H, W), mode='bilinear', align_corners=True)

            latent.add_(mixed * dynamic_scale)
        return latent

    def apply_vae_material_surgery(self, latent: torch.Tensor, injection_payload: Dict[str, Any],
                                   base_sampler_scale: float = 0.3) -> torch.Tensor:
        with torch.no_grad():
            block_B = injection_payload["block_B"]
            block_C = injection_payload["block_C"]
            B, C_latent, H, W = latent.shape

            if C_latent == 16:
                c_sampler, c_vae = 16, 4
                target_slice = latent[:, 12:16, :, :]
            elif C_latent == 4:
                c_sampler, c_vae = 4, 1
                target_slice = latent[:, 3:4, :, :]
            else:
                c_vae = min(16, C_latent)
                c_sampler = C_latent
                target_slice = latent[:, -c_vae:, :, :]

            auto_gain = (c_sampler / (2.0 * self.alpha * c_vae)) * base_sampler_scale
            auto_gain = min(auto_gain, 2.2)

            dev_B = block_B.to(latent.device, dtype=latent.dtype)
            dev_C = block_C.to(latent.device, dtype=latent.dtype)

            mixed_vae = torch.zeros(B, c_vae, dev_B.shape[2], dev_B.shape[3],
                                    device=latent.device, dtype=latent.dtype)
            for c in range(c_vae):
                if c % 2 == 0:
                    mixed_vae[:, c, :, :] = dev_B[:, -(c // 2 + 1), :, :]
                else:
                    mixed_vae[:, c, :, :] = dev_C[:, c // 2, :, :]

            if mixed_vae.shape[2:] != (H, W):
                mixed_vae = F.interpolate(mixed_vae, size=(H, W), mode='bilinear', align_corners=True)

            target_slice.add_(mixed_vae * auto_gain)
        return latent

    def _get_dynamic_scale(self, dt: float) -> float:
        if dt <= 0:
            return self.base_influence_scale
        return min(self.base_influence_scale * (dt / self.dt_ref), self.base_influence_scale * 3.0)

    def clear_all_cache(self) -> None:
        if self.cache_dir and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.disk_paths.clear()
        self._memory_buffer = None
        os.makedirs(self.cache_dir, exist_ok=True)

    def shutdown(self):
        if hasattr(self, '_executor'):
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

    def __del__(self):
        self.shutdown()