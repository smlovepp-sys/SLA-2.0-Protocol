# P4DecoderSurgery.py (最终纯净版 - 移除所有物理注入，仅保留解码与锐化 + 张量完整性检查)
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_management
from typing import Dict, Optional, Any

print(">>> P4DecoderSurgery 纯净解码版已激活 <<<")

class P4DecoderSurgery(nn.Module):
    def __init__(self, vae, adaptive_scale_gain: float = 0.65, guide_sharpness: float = 2.0):
        super().__init__()
        self.vae = vae
        self.adaptive_scale_gain = adaptive_scale_gain
        self.guide_sharpness = guide_sharpness
        print(f"[P4Decoder] 纯净解码模式 | 仅保留锐化与时序平滑")

        # 高频提取卷积核（用于锐化）
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('laplacian_kernel', laplacian)
        self.register_buffer('sobel_x_kernel', sobel_x)
        self.register_buffer('sobel_y_kernel', sobel_y)

    @staticmethod
    def check_tensor_integrity(tensor: torch.Tensor, name: str, threshold: float = 10.0):
        """检查张量中是否存在 NaN/Inf 或极端值，输出诊断信息"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"[CRITICAL] {name} 出现 NaN/Inf!")
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        if abs(max_val) > threshold or abs(min_val) > threshold:
            print(f"[WARN] {name} 数值异常: Max={max_val:.4f}, Min={min_val:.4f} | 可能发生梯度爆炸")
        if tensor.dim() >= 3:  # 至少是 3D (C,H,W) 才检查通道均值
            # 对于 4D [B,C,H,W] 或 5D [B,C,T,H,W] 取空间均值
            if tensor.dim() == 4:
                channel_means = tensor.abs().mean(dim=(0, 2, 3))
            elif tensor.dim() == 5:
                channel_means = tensor.abs().mean(dim=(0, 2, 3, 4))
            else:
                channel_means = None
            if channel_means is not None and channel_means.max() > threshold / 2:
                idx = torch.argmax(channel_means)
                print(f"[DEBUG] {name} 中第 {idx} 通道均值过高，可能是过曝源头")

    def _get_native_pytorch_model(self, vae_wrapper):
        if hasattr(vae_wrapper, "first_stage_model"):
            vae_wrapper = vae_wrapper.first_stage_model
        if hasattr(vae_wrapper, "model") and hasattr(vae_wrapper.model, "decode"):
            return vae_wrapper.model
        if hasattr(vae_wrapper, "patcher") and hasattr(vae_wrapper.patcher, "model"):
            return vae_wrapper.patcher.model
        curr = vae_wrapper
        while hasattr(curr, "model") and hasattr(curr.model, "decode"):
            curr = curr.model
        return curr

    def decode_with_p3_sync(
            self,
            latent: torch.Tensor,
            budgets: Dict[str, torch.Tensor],
            main_controller: Optional[Any] = None,
            memory_bank: Optional[Any] = None,
            frame_idx: int = 0,
            global_strength: Optional[float] = None,
            use_auto_gain: bool = True,
            extra_physics: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        纯净解码流程：仅执行 VAE 解码、可选锐化与时序平滑，不再进行任何物理注入。
        """
        device = latent.device
        dtype = latent.dtype

        # 入口检查
        self.check_tensor_integrity(latent, f"latent_in frame {frame_idx}")

        # 保持维度兼容
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)
        T_latent = latent.shape[2]

        # 打印基本信息
        lat_std_val = latent.std()
        if frame_idx % 4 == 0 or frame_idx == 0:
            print(f"  [P4Decoder:Surgery] 帧 {frame_idx:02d} | 潜空间Std: {lat_std_val.item():.3f} | 纯净解码模式")

        # ---------- VAE 解码 ----------
        with torch.no_grad():
            native_model = self._get_native_pytorch_model(self.vae)
            target_device = comfy.model_management.get_torch_device()

            if next(native_model.parameters()).device != target_device:
                native_model.to(target_device)

            target_dtype = next(native_model.parameters()).dtype

            # 根据 VAE 期望的输入维度调整 latent
            first_conv = next((m for m in native_model.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))), None)

            vae_in_latent = latent.clone()
            if isinstance(first_conv, nn.Conv2d) and vae_in_latent.dim() == 5:
                vae_in_latent = vae_in_latent.squeeze(2)
            elif isinstance(first_conv, nn.Conv3d) and vae_in_latent.dim() == 4:
                vae_in_latent = vae_in_latent.unsqueeze(2)

            vae_in_latent = vae_in_latent.to(device=target_device, dtype=target_dtype)
            scaling_factor = getattr(self.vae, 'scaling_factor', getattr(native_model, 'scaling_factor', 1.0))

            # 多通路 VAE 解码
            output = None

            # 通路 1：WanVideoVAE 等携带 device 参数
            try:
                output = self.vae.decode(vae_in_latent, device=vae_in_latent.device)
            except TypeError:
                pass

            # 通路 2：标准 VAE 解码
            if output is None:
                try:
                    output = self.vae.decode(vae_in_latent)
                except Exception:
                    pass

            # 通路 3：降级到底层原生模型
            if output is None:
                raw_model = getattr(self.vae, "first_stage_model", getattr(self.vae, "vae", self.vae))
                if raw_model is not None:
                    if not hasattr(raw_model, '_feat_map'):
                        setattr(raw_model, '_feat_map', None)
                    import inspect
                    sig = inspect.signature(raw_model.decode)
                    if 'device' in sig.parameters:
                        output = raw_model.decode(vae_in_latent / scaling_factor, device=vae_in_latent.device)
                    else:
                        output = raw_model.decode(vae_in_latent / scaling_factor)
                else:
                    raise RuntimeError("[P4/Surgery] 致命错误：无法定位到任何有效的 VAE 解码模型实例。")

            if isinstance(output, (tuple, list)):
                output = output[0]

            # 进一步解包
            if isinstance(output, list):
                decoded = output[0]
            elif hasattr(output, 'sample'):
                decoded = output.sample
            else:
                decoded = output
            if isinstance(decoded, list):
                decoded = decoded[0]

        # 提取需要的时序帧
        if decoded.dim() == 5 and decoded.shape[2] > 1:
            decoded = decoded[:, :, -1:, :, :]
        if decoded.dim() == 5 and decoded.shape[2] == 1:
            decoded = decoded.squeeze(2)

        self.check_tensor_integrity(decoded, f"decoded_raw frame {frame_idx}")

        # 可选锐化
        if self.guide_sharpness > 0:
            decoded = self._sharpen(decoded, self.guide_sharpness, detail_mask=None)

        self.check_tensor_integrity(decoded, f"decoded_sharpened frame {frame_idx}")

        # 时序平滑（如果提供了 memory_bank）
        if memory_bank is not None:
            gray = decoded.mean(dim=1, keepdim=True) if decoded.shape[1] > 1 else decoded
            dx = gray[:, :, :, :-1] - gray[:, :, :, 1:]
            dy = gray[:, :, :-1, :] - gray[:, :, 1:, :]
            energy_map = torch.sqrt(dx.pow(2).mean(dim=-1, keepdim=True) + dy.pow(2).mean(dim=-2, keepdim=True))
            energy_map = F.pad(energy_map, (0, 1, 0, 1), mode='replicate')
            memory_bank.update(decoded, energy_map=energy_map)
            decoded = memory_bank.get_smoothed()
            self.check_tensor_integrity(decoded, f"decoded_smoothed frame {frame_idx}")

        return decoded

    def decode_with_fsync(self, latent, budgets, **kwargs):
        return self.decode_with_p3_sync(
            latent=latent,
            budgets=budgets,
            main_controller=kwargs.get("main_controller"),
            memory_bank=kwargs.get("memory_bank"),
            frame_idx=kwargs.get("frame_idx", 0),
            global_strength=kwargs.get("global_strength"),
            use_auto_gain=kwargs.get("use_auto_gain", True),
            extra_physics=kwargs.get("extra_physics"),
        )

    def _sharpen(self, image: torch.Tensor, strength: float = 2.0, detail_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        安全锐化：仅在边缘处应用拉普拉斯算子，防止平坦区域噪点放大。
        """
        if image.dim() == 3: image = image.unsqueeze(0)
        orig_shape = image.shape

        # 处理 5D 张量 (B, C, T, H, W)
        if image.dim() == 5:
            B, C, T, H, W = orig_shape
            image_4d = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            is_5d = True
        else:
            B, C, H, W = orig_shape
            T = 1
            image_4d = image
            is_5d = False

        # 1. 如果没有传入 mask，则基于 Sobel 生成一个边缘掩码
        if detail_mask is None:
            gray = image_4d.mean(dim=1, keepdim=True)
            grad_x = F.conv2d(gray, self.sobel_x_kernel.to(image_4d.device, image_4d.dtype), padding=1)
            grad_y = F.conv2d(gray, self.sobel_y_kernel.to(image_4d.device, image_4d.dtype), padding=1)
            detail_mask = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            # 归一化并增强对比度，只保留强边缘
            detail_mask = torch.clamp(detail_mask * 5.0, 0.0, 1.0)

        # 2. 拉普拉斯锐化
        laplacian_k = self.laplacian_kernel.to(dtype=image_4d.dtype, device=image_4d.device)
        curr_c = image_4d.shape[1]
        high_freq = F.conv2d(image_4d, laplacian_k.repeat(curr_c, 1, 1, 1), padding=1, groups=curr_c)

        # 3. 混合：只有在 detail_mask 处应用 high_freq
        sharpened_image = image_4d + (high_freq * detail_mask * strength * 0.05)

        # 恢复维度
        if is_5d:
            result = sharpened_image.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        else:
            result = sharpened_image

        return result