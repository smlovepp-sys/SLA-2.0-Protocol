# P4Decoder.py (终极救急版 - 暴力压制过曝 -> 安全 clamp 版 + 张量完整性检查)
import torch
import torch.nn.functional as F
import gc
from typing import Dict, Any, Optional, Tuple
from safetensors.torch import load_file
from .P4DecoderSurgery import P4DecoderSurgery
from .P4ColorModulator import P4ColorModulator
from .P4MemoryBank import P4MemoryBank
from .p4_prefetch_scheduler import P4PrefetchScheduler


class P4Decoder:
    def __init__(self,
                 vae: torch.nn.Module,
                 p3_payload: Dict[str, Any],
                 adaptive_scale_gain: float = 0.15,
                 guide_sharpness: float = 2.0,
                 enable_prefetch: bool = True,
                 target_device: str = "cuda"):
        self.vae = vae
        self.device = torch.device(target_device if torch.cuda.is_available() else "cpu")
        self.p3_payload = p3_payload
        self.adaptive_scale_gain = adaptive_scale_gain
        self.guide_sharpness = guide_sharpness

        # 初始化组件
        self.surgery = P4DecoderSurgery(vae, p3_payload, adaptive_scale_gain)
        self.color_mod = P4ColorModulator()
        self.memory_bank = P4MemoryBank()

        # 预取调度器
        self.prefetcher = None
        if enable_prefetch and 'frame_paths' in p3_payload:
            self.prefetcher = P4PrefetchScheduler(p3_payload['frame_paths'], device=self.device)

        print(f"[P4Decoder] 初始化完成 | 纯净解码模式 | 基础自适应增益: {adaptive_scale_gain}")

    @staticmethod
    def check_tensor_integrity(tensor: torch.Tensor, name: str, threshold: float = 10.0):
        """
        如果张量中有极端值，立即抛出详细诊断信息。
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"[CRITICAL] {name} 出现 NaN/Inf!")

        max_val = tensor.max().item()
        min_val = tensor.min().item()

        if abs(max_val) > threshold or abs(min_val) > threshold:
            print(f"[WARN] {name} 数值异常: Max={max_val:.4f}, Min={min_val:.4f} | 此时可能发生了梯度爆炸")

        # 记录通道分布，发现哪个通道“炸”了
        if tensor.dim() == 4:
            channel_means = tensor.abs().mean(dim=(0, 2, 3))
            if channel_means.max() > threshold / 2:
                idx = torch.argmax(channel_means)
                print(f"[DEBUG] {name} 中第 {idx} 通道均值过高，可能是过曝源头")

    def decode_latents(self, latents: torch.Tensor, frame_indices: list = None) -> torch.Tensor:
        """
        核心解码流程：物理场注入 -> 手术修正 -> VAE解码
        """
        if frame_indices is None:
            frame_indices = list(range(latents.shape[2]))

        decoded_frames = []
        batch_size, channels, num_frames, h, w = latents.shape

        for i, frame_idx in enumerate(frame_indices):
            # 1. 获取当前帧的潜空间数据
            current_latent = latents[:, :, i:i+1, :, :].clone().to(self.device)

            # 2. 加载并处理物理场 (关键修复点)
            phys_field = self._load_and_process_physics(frame_idx, h, w)

            # 3. 注入物理场 (带强力衰减)
            injected_latent = self._inject_physics_safe(current_latent, phys_field)
            self.check_tensor_integrity(injected_latent, f"injected_latent frame {frame_idx}")

            # 4. 执行手术修正 (时序平滑 + 锐化)
            processed_latent = self.surgery.apply_surgery(injected_latent, frame_idx)
            self.check_tensor_integrity(processed_latent, f"processed_latent frame {frame_idx}")

            # 5. VAE 解码
            with torch.no_grad():
                # 确保输入 VAE 的数据在合理范围内，防止过曝
                safe_latent = torch.clamp(processed_latent, -1.5, 1.5)
                decoded_frame = self.vae.decode(safe_latent)

            self.check_tensor_integrity(decoded_frame, f"decoded_frame frame {frame_idx}")
            decoded_frames.append(decoded_frame.cpu())

            # 更新显存中的上一帧缓存
            self.memory_bank.update(processed_latent.detach())

        return torch.cat(decoded_frames, dim=0)

    def _load_and_process_physics(self, frame_idx: int, h: int, w: int) -> Dict[str, torch.Tensor]:
        """
        加载物理场并进行安全限幅处理（不再使用硬乘法压死信号）
        """
        physics_data = {}

        if self.prefetcher:
            raw_data = self.prefetcher.get_frame(frame_idx, blocking=True)
        elif 'physics_cache' in self.p3_payload:
            cache = self.p3_payload['physics_cache']
            if frame_idx < len(cache):
                raw_data = cache[frame_idx]
            else:
                raw_data = None
        else:
            raw_data = None

        if raw_data is not None:
            if isinstance(raw_data, dict):
                physics_data['geo'] = raw_data.get('geo', torch.zeros(1, 16, h, w)).to(self.device)
                physics_data['mat'] = raw_data.get('mat', torch.zeros(1, 16, h, w)).to(self.device)
            elif isinstance(raw_data, torch.Tensor):
                mid = raw_data.shape[1] // 2
                physics_data['geo'] = raw_data[:, :mid, :, :]
                physics_data['mat'] = raw_data[:, mid:, :, :]
        else:
            physics_data['geo'] = torch.zeros(1, 16, h, w, device=self.device)
            physics_data['mat'] = torch.zeros(1, 16, h, w, device=self.device)

        # 安全限幅
        physics_data['mat'] = torch.clamp(physics_data['mat'], 0.0, 1.0)
        physics_data['geo'] = torch.clamp(physics_data['geo'], 0.0, 1.0)

        self.check_tensor_integrity(physics_data['geo'], f"physics_geo frame {frame_idx}")
        self.check_tensor_integrity(physics_data['mat'], f"physics_mat frame {frame_idx}")

        return physics_data

    def _inject_physics_safe(self, latent: torch.Tensor, physics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        安全的物理场注入逻辑
        """
        geo = physics['geo'].unsqueeze(2)  # [1, 16, 1, H, W]
        mat = physics['mat'].unsqueeze(2)

        result = latent + geo + mat
        return result

    def close(self):
        """清理资源"""
        if self.prefetcher:
            self.prefetcher.stop()
        del self.surgery
        del self.color_mod
        del self.memory_bank
        gc.collect()
        torch.cuda.empty_cache()