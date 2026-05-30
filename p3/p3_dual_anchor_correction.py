# p3_dual_anchor_correction.py (防御修复版：兼容快照结构与通道数)
import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Any, Tuple

class P3DualAnchorCorrection:
    def __init__(self, mae_threshold: float = 0.15, ssim_threshold: float = 0.85,
                 blend_lambda: float = 0.3):
        self.mae_threshold = mae_threshold
        self.ssim_threshold = ssim_threshold
        self.blend_lambda = blend_lambda

    def check_and_correct(self, snapshots: List[Dict[str, Any]],
                          end_frame: torch.Tensor) -> Tuple[List[Dict[str, Any]], bool, float]:
        """
        snapshots: 演化生成的快照列表，最后一个为最后一帧
        end_frame: P1提供的结束帧 [1, C, H, W]（通常C=3或4）
        返回: (修正后的快照列表, 是否修正, 锚定MAE值)
        """
        # --- 防御：空快照或缺少关键字段时直接返回 ---
        if not snapshots:
            return snapshots, False, 0.0

        last_snap = snapshots[-1]
        pred = last_snap.get('latent_patch')   # 当前快照中不存在此字段，保留以备将来使用

        # 若没有 latent_patch，无法执行基于潜在空间的修正，安全退出
        if pred is None:
            # 可在此处计算一个替代 MAE（如基于应力场），但为保持简单返回 0
            return snapshots, False, 0.0

        # --- 防御：处理 end_frame 通道数不足 48 的情况 ---
        if end_frame.dim() != 4:
            return snapshots, False, 0.0

        C_end = end_frame.shape[1]
        if C_end < 3:
            return snapshots, False, 0.0

        # 使用前3通道作为正常信息，如果 end_frame 通道数不够48则放弃撕裂通道比较
        pred_normal = pred[:, :3, :, :]
        target_normal = end_frame[:, :3, :, :]          # 至少3通道

        # 设备同步：确保在同一设备上计算
        if pred.device != target_normal.device:
            target_normal = target_normal.to(pred.device)

        mae_normal = F.l1_loss(pred_normal, target_normal).item()
        ssim_val = self._compute_ssim(pred_normal, target_normal)

        need_correction = (mae_normal > self.mae_threshold) or (ssim_val < self.ssim_threshold)
        if not need_correction:
            return snapshots, False, mae_normal

        # --- 执行修正（仅当 pred 和 end_frame 均为 48 通道时完整进行，否则降级）---
        # 如果 end_frame 通道数不足48，则无法对撕裂通道进行修正，此时只混合前3通道
        corrected_snapshots = list(snapshots)
        corrected_last = last_snap.copy()

        # 仅当通道数足够时才进行全通道混合，否则只混合前几个通道（保持其他通道不变）
        if C_end >= pred.shape[1]:   # 假设 pred 是 48ch，end_frame 也是 48ch 或更多
            blended_patch = (1 - self.blend_lambda) * pred + self.blend_lambda * end_frame
        else:
            # 降级：只混合前 C_end 个通道，其余保持 pred 原值
            blended_patch = pred.clone()
            blended_patch[:, :C_end, :, :] = (1 - self.blend_lambda) * pred[:, :C_end, :, :] + \
                                             self.blend_lambda * end_frame
        corrected_last['latent_patch'] = blended_patch
        corrected_snapshots[-1] = corrected_last

        # 余弦衰减分配至最后20%帧
        total = len(snapshots)
        N = max(3, int(total * 0.2))
        for i in range(1, N + 1):
            idx = total - 1 - i
            if idx < 0:
                break
            weight = 0.5 * (1 + math.cos(math.pi * i / N)) * self.blend_lambda
            snap = snapshots[idx].copy()
            orig = snap.get('latent_patch')
            if orig is not None:
                if C_end >= orig.shape[1]:
                    blended = (1 - weight) * orig + weight * end_frame
                else:
                    blended = orig.clone()
                    blended[:, :C_end, :, :] = (1 - weight) * orig[:, :C_end, :, :] + \
                                               weight * end_frame
                snap['latent_patch'] = blended
                corrected_snapshots[idx] = snap

        return corrected_snapshots, True, mae_normal

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        # 确保两张图在同一设备上
        if img1.device != img2.device:
            img2 = img2.to(img1.device)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu1 = F.avg_pool2d(img1, window_size, 1, 0)
        mu2 = F.avg_pool2d(img2, window_size, 1, 0)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, 0) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, 0) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, 0) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        return ssim_map.mean().item()