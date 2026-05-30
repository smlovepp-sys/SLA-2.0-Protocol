# p4_comfy_nodes.py (应力增益修复 + VAE 输出规范化 + 禁用不稳定的 min-max 拉伸)
import torch
import torch.nn.functional as F
import comfy.sample
import comfy.utils
import comfy.model_management
import os
import math
from safetensors.torch import load_file
import gc

from .P4DecoderSurgery import P4DecoderSurgery
from .P4MemoryBank import P4MemoryBank

class P4PhysicalSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("*",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "p3_payload": ("P3_PAYLOAD",),
                "alpha": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "antialias_input": ("BOOLEAN", {"default": True}),
                "auto_batch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "P4 Physics"

    @staticmethod
    def _tensor_stats(tensor, name="tensor"):
        if tensor is None or tensor.numel() == 0:
            return f"{name}: EMPTY"
        t = tensor.float()
        return (f"{name}: shape={list(tensor.shape)}, "
                f"mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
                f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
                f"NaN={torch.isnan(t).any().item()}")

    def _safe_to_image_format(self, tensor):
        # 保留原有逻辑，但移除不稳定的 min-max 自动拉伸
        if tensor.dim() == 4 and tensor.shape[-1] == 3:
            tensor = tensor.float().cpu()
        elif tensor.dim() == 4 and tensor.shape[1] == 3:
            tensor = tensor.permute(0, 2, 3, 1).float().cpu()
        elif tensor.dim() == 5:
            if tensor.shape[-1] == 3:
                tensor = tensor.reshape(-1, tensor.shape[-3], tensor.shape[-2], 3).float().cpu()
            elif tensor.shape[1] == 3:
                tensor = tensor.squeeze(2).permute(0, 2, 3, 1).float().cpu()
        else:
            shape = list(tensor.shape)
            channel_dim = None
            for i, s in enumerate(shape):
                if s == 3: channel_dim = i; break
            if channel_dim is not None:
                dims = list(range(tensor.dim()))
                dims.pop(channel_dim)
                dims.append(channel_dim)
                tensor = tensor.permute(*dims).contiguous()
                while tensor.dim() > 4:
                    for i in range(1, tensor.dim() - 3):
                        if tensor.shape[i] == 1:
                            tensor = tensor.squeeze(i)
                            break
                    else:
                        break
                if tensor.dim() == 5:
                    tensor = tensor.reshape(-1, tensor.shape[-3], tensor.shape[-2], 3)
            tensor = tensor.float().cpu()

        # 移除不稳定的 min-max 自动拉伸，改为直接 clamp
        return tensor.clamp(0.0, 1.0)

    def _enhance_color(self, rgb_tensor, strength=0.1):
        input_hwc = (rgb_tensor.shape[-1] == 3)
        if input_hwc:
            rgb = rgb_tensor.permute(0, 3, 1, 2).contiguous()
        else:
            rgb = rgb_tensor

        contrast = 1.0 + strength * 0.5
        rgb = 0.5 + (rgb - 0.5) * contrast

        gray = rgb.mean(dim=1, keepdim=True)
        saturation = 1.0 + strength * 0.6
        rgb = gray + (rgb - gray) * saturation

        rgb = rgb.clamp(0.0, 1.0)

        if input_hwc:
            return rgb.permute(0, 2, 3, 1).contiguous()
        return rgb

    def sample(self, model, vae, positive, negative, latent, p3_payload, alpha, antialias_input, auto_batch):
        print("\n[P4:Sampler] ======= 物理采样节点激活 (应力增强 + 亮度修复) =======")

        frame_paths = p3_payload.get("frame_file_paths", [])
        metadata_list = p3_payload.get("frame_metadata_list", [])
        T = len(frame_paths)
        if T == 0:
            raise ValueError("[P4:Sampler] 无帧路径")

        main_controller = p3_payload.get("main_controller", None)
        if main_controller is None:
            raise RuntimeError("[P4:Sampler] 缺失 main_controller")

        diff_params = p3_payload.get("p2_diffusion_params", {})
        steps = diff_params.get("num_inference_steps", 20)
        cfg = diff_params.get("cfg_scale", 7.5)
        seed = diff_params.get("seed", 42)
        sampler_name = diff_params.get("sampler_type", "euler")
        scheduler = diff_params.get("scheduler", "normal")
        denoise = 1.0

        latent_tensor = latent["samples"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_tensor = latent_tensor.to(device)
        if latent_tensor.dim() == 4:
            latent_tensor = latent_tensor.unsqueeze(2)
        B, C, T_lat, H_lat, W_lat = latent_tensor.shape
        print(f"[P4:Sampler] 潜空间形状: {latent_tensor.shape}")

        memory_bank = P4MemoryBank(base_beta=0.5, energy_sensitivity=0.85)
        decoder_surgery = P4DecoderSurgery(vae, adaptive_scale_gain=0.35, guide_sharpness=2.0)
        latent_noise = comfy.sample.prepare_noise(latent_tensor, seed)

        callback_stats = []

        def physics_callback(step, x0, x, total_steps):
            with torch.no_grad():
                if step != 0: return x
                memory_bank.set_progress(step / max(total_steps, 1))
                print(f"[P4:Callback] 注入物理场")
                for t in range(min(T, x.shape[2])):
                    phys_cpu = main_controller.get_frame_slice(t, slice_idx=0, device="cpu")
                    if phys_cpu.dim() == 3:
                        phys_cpu = phys_cpu.unsqueeze(0).unsqueeze(2)
                    elif phys_cpu.dim() == 4:
                        phys_cpu = phys_cpu.unsqueeze(2)

                    phys_frame = phys_cpu.to(device=x.device, dtype=x.dtype)
                    phys_frame = F.interpolate(phys_frame.squeeze(2), size=(x.shape[3], x.shape[4]), mode='bilinear', align_corners=False).unsqueeze(2)

                    # 诊断打印（保留）
                    print(self._tensor_stats(phys_frame, f"  Frame {t} phys_frame"))
                    phys_geo = phys_frame[:, :16]
                    phys_mat = phys_frame[:, 16:32]
                    phys_det = phys_frame[:, 32:48]
                    print(f"    geo  (0-15): mean={phys_geo.mean().item():.4f}, std={phys_geo.std().item():.4f}, min={phys_geo.min().item():.4f}, max={phys_geo.max().item():.4f}")
                    print(f"    mat  (16-31): mean={phys_mat.mean().item():.4f}, std={phys_mat.std().item():.4f}, min={phys_mat.min().item():.4f}, max={phys_mat.max().item():.4f}")
                    print(f"    det  (32-47): mean={phys_det.mean().item():.4f}, std={phys_det.std().item():.4f}, min={phys_det.min().item():.4f}, max={phys_det.max().item():.4f}")

                    # 🔧 关键修复：提高应力注入增益
                    phys_inject = phys_frame[:, :16, :, :, :] * 2.0  # 原来 0.15 -> 2.0

                    stress_mean = phys_frame[:, 0:1, :, :, :].mean().item()
                    wetness_mean = phys_frame[:, 16:17, :, :, :].mean().item()
                    callback_stats.append({
                        "frame": t,
                        "stress_mean": stress_mean,
                        "wetness_mean": wetness_mean
                    })

                    current_slice = x[:, :, t:t+1, :, :]
                    x[:, :, t:t+1, :, :] = current_slice + phys_inject
                    print(f"    注入后潜变量变化: mean={phys_inject.mean().item():.6f}, std={phys_inject.std().item():.6f}")
                    del phys_frame, phys_cpu, phys_inject

                torch.cuda.empty_cache()
            return x

        ret = comfy.sample.sample(
            model=model, noise=latent_noise, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative,
            latent_image=latent_tensor, denoise=denoise, seed=seed,
            callback=physics_callback
        )
        result_5d = ret[0] if isinstance(ret, tuple) else ret

        print(f"[P4:Sampler] 采样完成，形状: {result_5d.shape}")

        if callback_stats:
            print("\n===== [采样阶段物理注入汇总] =====")
            print(f"{'帧':>4} {'应力均值':>10} {'湿润均值':>10}")
            for s in callback_stats:
                print(f"{s['frame']:4d} {s['stress_mean']:10.4f} {s['wetness_mean']:10.4f}")
            print("===================================\n")

        if result_5d.dim() == 4:
            result_5d = result_5d.unsqueeze(2)
        B_res, C_res, T_res, H_res, W_res = result_5d.shape

        decoded_frames = []
        decode_stats = []

        for t in range(T_res):
            if result_5d.dim() == 5:
                latent_frame = result_5d[:, :, t:t+1, :, :]
            else:
                latent_frame = result_5d[t:t+1]

            latent_frame = latent_frame.to(device)
            memory_bank.lock_target_std(latent_frame)

            path = frame_paths[t] if t < T else frame_paths[-1]
            if not os.path.exists(path):
                raise FileNotFoundError(f"物理载荷文件缺失: {path}")

            raw_data = load_file(path)
            block_A = raw_data["block_A"].to(device=device, dtype=latent_tensor.dtype, non_blocking=True)
            block_B = raw_data["block_B"].to(device=device, dtype=latent_tensor.dtype, non_blocking=True)
            block_C = raw_data["block_C"].to(device=device, dtype=latent_tensor.dtype, non_blocking=True)

            block_A = block_A.unsqueeze(2)
            block_B = block_B.unsqueeze(2)
            block_C = block_C.unsqueeze(2)

            full_48ch = torch.cat([block_A, block_B, block_C], dim=1)
            if full_48ch.shape[-2:] != (H_res, W_res):
                full_48ch = F.interpolate(full_48ch.squeeze(2), size=(H_res, W_res), mode='bilinear', align_corners=False).unsqueeze(2)

            print(self._tensor_stats(full_48ch, f"  Frame {t} full_48ch"))

            budgets = {
                "geo": full_48ch[:, 0:16, :, :, :],
                "mat": full_48ch[:, 16:32, :, :, :],
                "det": full_48ch[:, 32:48, :, :, :],
            }

            print(f"    geo: mean={budgets['geo'].mean().item():.4f}, std={budgets['geo'].std().item():.4f}, min={budgets['geo'].min().item():.4f}, max={budgets['geo'].max().item():.4f}")
            print(f"    mat: mean={budgets['mat'].mean().item():.4f}, std={budgets['mat'].std().item():.4f}, min={budgets['mat'].min().item():.4f}, max={budgets['mat'].max().item():.4f}")
            print(f"    det: mean={budgets['det'].mean().item():.4f}, std={budgets['det'].std().item():.4f}, min={budgets['det'].min().item():.4f}, max={budgets['det'].max().item():.4f}")

            # 调用纯净 VAE 解码
            img_frame = decoder_surgery.decode_with_fsync(
                latent_frame,
                budgets,
                memory_bank=memory_bank,
                frame_idx=t
            )

            # ======= 维度对齐逻辑 =======
            if img_frame.dim() == 5:
                B, C, T_img, H, W = img_frame.shape
                if C in [3, 4]:
                    img_frame = img_frame.permute(0, 2, 3, 4, 1).contiguous()
                    img_frame = img_frame.view(-1, H, W, C)
                else:
                    img_frame = img_frame.squeeze(2)

            if img_frame.dim() == 4:
                if img_frame.shape[1] == 3 or img_frame.shape[1] == 4:
                    img_frame = img_frame.permute(0, 2, 3, 1).contiguous()
                elif img_frame.shape[-1] == 3:
                    pass
                else:
                    img_frame = img_frame.contiguous()

            if img_frame.shape[-1] == 4:
                img_frame = img_frame[:, :, :, :3]

            # 🔧 关键修复：WAN21 VAE 输出规范化
            if img_frame.mean() < 0.3:
                img_frame = (img_frame + 1.0) / 2.0
            img_frame = img_frame.clamp(0.0, 1.0)

            # 颜色增强：默认关闭
            enhance_strength = 0.0
            if enhance_strength > 0:
                img_frame = self._enhance_color(img_frame, strength=enhance_strength)

            final_frame = self._safe_to_image_format(img_frame)
            print(self._tensor_stats(final_frame, f"  Frame {t} final_frame"))
            decoded_frames.append(final_frame)

            # 记录统计
            geo_mean = budgets["geo"].squeeze(2).mean().item()
            mat_mean = budgets["mat"].squeeze(2).mean().item()
            decode_stats.append({
                "frame": t,
                "geo_mean": geo_mean,
                "mat_mean": mat_mean,
                "out_mean": final_frame.mean().item(),
                "latent_std": latent_frame.std().item(),
            })

            # 显式释放资源
            del full_48ch, budgets, img_frame, latent_frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_output = torch.cat(decoded_frames, dim=0)

        if decode_stats:
            print("\n============= [解码阶段统计] =============")
            print(f"{'帧':>4} {'geo原均值':>10} {'mat原均值':>10} {'输出均值':>10} {'Std':>8}")
            for s in decode_stats:
                print(f"{s['frame']:4d} {s['geo_mean']:10.4f} {s['mat_mean']:10.4f} {s['out_mean']:10.3f} {s['latent_std']:8.3f}")
            print("===========================================\n")

        memory_bank.clear_temporal_memory()
        comfy.model_management.soft_empty_cache()
        gc.collect()

        print("[P4:Sampler] ======= 完成 =======\n")
        return (final_output,)


NODE_CLASS_MAPPINGS = {"P4PhysicalSampler": P4PhysicalSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"P4PhysicalSampler": "P4 Physical Sampler (应力增强+亮度修复)"}