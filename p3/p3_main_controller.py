# p3_main_controller.py (智能实体识别版：修复 global_physics 优先读取 + 应力场编码)
import os
import json
import gc
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from safetensors.torch import save_file

from .p3_global_mario_backbone import P3GlobalMarioBackbone
from .p3_physics_engine import P3PhysicsEngine
from .p3_micro_impact import P3MicroImpact
from .p3_micro_detail_tear import P3MicroDetailTear
from .p3_temporal_vae_hijacker import P3TemporalVAEHijacker
from .p3_cipher_engine import P3CipherEngine
from .p3_neural_sync_bridge import P3NeuralSyncBridge
from .p3_superiority_engine import P3SuperiorityEngine
from .p3_material_memory import P3MaterialMemory
from .p3_kinetic_coupling import P3KineticCoupling
from .p3_topology_adapter import P3TopologyAdapter
from .p3_stress_activator import P3StressActivator
from .p3_dynamic_feedback import P3DynamicFeedback
from .p3_slot_manager import P3SlotManager
from .p3_memory_encoder import P3MemoryEncoder
from .p3_golden_evaluator import P3GoldenEvaluator
from .p3_dual_anchor_correction import P3DualAnchorCorrection


class P3Controller:
    def __init__(self,
                 fps: float = 30.0,
                 alpha: float = 0.15,
                 vae_mode: str = "auto",
                 storage_mode: str = "disk",
                 cache_dir: Optional[str] = None,
                 fallback_mode: str = "repeat_last",
                 slot_cache_dir: Optional[str] = None,
                 enable_neural_sync: bool = False,
                 neural_sync_strength: float = 0.1,
                 preserve_p1_details: bool = False):
        self.fps = fps
        self.alpha = alpha
        self.vae_mode = vae_mode
        self.storage_mode = storage_mode
        self.fallback_mode = fallback_mode
        self.enable_neural_sync = enable_neural_sync
        self.neural_sync_strength = neural_sync_strength
        self.preserve_p1_details = preserve_p1_details

        if cache_dir is None:
            import tempfile
            cache_dir = tempfile.mkdtemp(prefix="p3_main_")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        if slot_cache_dir is None:
            slot_cache_dir = os.path.join(self.cache_dir, "slots")
        self.slot_cache_dir = slot_cache_dir
        os.makedirs(self.slot_cache_dir, exist_ok=True)

        self.backbone: Optional[P3GlobalMarioBackbone] = None
        self.engine: Optional[P3PhysicsEngine] = None
        self.superior_engine: Optional[P3SuperiorityEngine] = None
        self.memory: Optional[P3MaterialMemory] = None
        self.micro_impact: Optional[P3MicroImpact] = None
        self.micro_tear: Optional[P3MicroDetailTear] = None
        self.kinetic: Optional[P3KineticCoupling] = None
        self.topology: Optional[P3TopologyAdapter] = None
        self.stress_activator: Optional[P3StressActivator] = None
        self.dynamic_feedback: Optional[P3DynamicFeedback] = None
        self.hijacker: Optional[P3TemporalVAEHijacker] = None
        self.cipher_engine: Optional[P3CipherEngine] = None
        self.sync_bridge: Optional[P3NeuralSyncBridge] = None

        self.slot_manager: Optional[P3SlotManager] = None
        self.memory_encoder: Optional[P3MemoryEncoder] = None
        self.golden_evaluator: Optional[P3GoldenEvaluator] = None
        self.anchor_corrector: Optional[P3DualAnchorCorrection] = None

        self._first_frame_4ch: Optional[torch.Tensor] = None
        self._first_frame_metadata: Optional[Dict[str, Any]] = None

        # 统一物理参数键（与 P1 输出完全对齐）
        self.GLOBAL_DEFAULT_PHYSICS = {
            "stiffness": 0.5,
            "friction": 0.5,
            "elasticity": 0.5,          # 原 elastic_recovery
            "wetness": 0.0,
            "tear_intensity": 0.0,
            "tension": 0.0,
            "body_type": 0.5,
            "ior": 1.33,                # 保留旧的折射率
            "depth_rank": 0
        }
        self.asset_configs: Dict[str, Dict[str, float]] = {}
        self.dressing_action: Optional[str] = None
        self.current_manifest: Dict[str, Any] = {}
        self.current_p3_payload: Dict[str, Any] = {}
        self.current_p2_package: Dict[str, Any] = {}

        self.lat_raw_info = {"orig_height": 0, "orig_width": 0, "total_frames": 1}
        self._internal_8x_h = 0
        self._internal_8x_w = 0
        self.detected_channels = 16

        self.physics_momentum_cache: Dict[str, Dict[str, float]] = {}

        print(f"[P3:Controller] 初始化就绪 | FPS: {fps} | VAE: {vae_mode} | NeuralSync: {'ON' if enable_neural_sync else 'OFF'} | PreserveP1Details: {'ON' if preserve_p1_details else 'OFF'}")

    def _init_modules(self, p1_manifest: Dict[str, Any], seed: int = 42) -> None:
        if self.backbone is not None:
            return
        self.backbone = P3GlobalMarioBackbone(
            world_size=(1.0, 1.0),
            tile_resolution=(self._internal_8x_h, self._internal_8x_w)
        )
        self.engine = P3PhysicsEngine()
        self.superior_engine = P3SuperiorityEngine()
        self.memory = P3MaterialMemory()
        self.micro_impact = P3MicroImpact()
        self.micro_tear = P3MicroDetailTear()
        self.kinetic = P3KineticCoupling()
        self.topology = P3TopologyAdapter()
        self.stress_activator = P3StressActivator()
        self.dynamic_feedback = P3DynamicFeedback(seed=seed)
        self.hijacker = P3TemporalVAEHijacker(
            fps=self.fps, alpha=self.alpha, vae_mode=self.vae_mode,
            cache_dir=self.cache_dir
        )
        self.cipher_engine = P3CipherEngine()
        self.sync_bridge = P3NeuralSyncBridge(physical_grid_size=32, latent_downscale_factor=8)
        self.slot_manager = P3SlotManager(cache_dir=self.slot_cache_dir)
        self.memory_encoder = P3MemoryEncoder(seed=seed)
        self.golden_evaluator = P3GoldenEvaluator()
        self.anchor_corrector = P3DualAnchorCorrection(
            mae_threshold=0.15, ssim_threshold=0.85, blend_lambda=0.3
        )
        if hasattr(self.backbone, 'init_grid'):
            self.backbone.init_grid(p1_manifest)

    def _compute_fingerprint(self, start_frame, end_frame, ciphers, base_physics):
        return self.slot_manager._compute_fingerprint(start_frame, end_frame, ciphers, base_physics)

    # 暗号平滑改为保留 P1 物理值（不再被暗号基线覆盖）
    def _apply_cipher_driven_physics(self, assets: Dict[str, Any], ciphers: List[str], beta: float = 0.8):
        print(f"[P3 Physics] 跳过暗号覆盖，保留 P1 提供的物理参数")
        for asset_id, asset_data in assets.items():
            if not isinstance(asset_data, dict):
                continue
            slots = asset_data.setdefault("slots", {})
            layer = slots.setdefault("slot_6_physics_layer", {})
            for key in self.GLOBAL_DEFAULT_PHYSICS:
                if key not in layer:
                    layer[key] = self.GLOBAL_DEFAULT_PHYSICS[key]

    def _process_single_frame(self, frame_idx, dt, p2_package, hand_grab_info=None, impacts=None):
        if frame_idx % 5 == 0:
            print(f"  [P3] 处理帧 {frame_idx}")
        assets = p2_package.get("assets", {})
        if isinstance(assets, dict):
            invalid_keys = [k for k, v in assets.items() if not isinstance(v, dict)]
            for k in invalid_keys:
                del assets[k]
        if self.superior_engine:
            p2_package = self.superior_engine.process_frame(p2_package, hand_grab_info=hand_grab_info, dt=dt)
        if self.memory and self.backbone:
            p2_package = self.memory.process_tiles(self.backbone, p2_package, dt=dt)
        if self.superior_engine is None and self.kinetic and hand_grab_info:
            p2_package = self.kinetic.process(p2_package, hand_grab_info)
        if self.dressing_action and self.topology:
            p2_package = self.topology.process(p2_package, action=self.dressing_action)
        if impacts and self.micro_impact:
            p2_package = self.micro_impact.process(p2_package, impacts, dt=dt)
        if self.micro_tear:
            p2_package = self.micro_tear.process_frame(p2_package, dt=dt)
        if self.stress_activator:
            p2_package = self.stress_activator.activate(p2_package)
        if self.dynamic_feedback:
            p2_package = self.dynamic_feedback.process(p2_package)
        if self.engine:
            p2_package = self.engine.process_physics(p2_package)
        return p2_package

    def _extract_physics_from_p2(self, p2_package):
        assets = p2_package.get("assets", {})
        extracted = {}
        for aid, adata in assets.items():
            if not isinstance(adata, dict):
                continue
            slots = adata.get("slots", {})
            physics = {}
            layer = slots.get("slot_6_physics_layer", {})
            for k in self.GLOBAL_DEFAULT_PHYSICS:
                physics[k] = layer.get(k, self.GLOBAL_DEFAULT_PHYSICS[k])
            extracted[aid] = physics
        return extracted

    def _build_snapshot_from_p2(self, frame_idx, p2_package):
        tile_info = p2_package.get("global_tile_info", {})
        max_stress = max((info.get("stress_mag", 0.0) for info in tile_info.values() if isinstance(info, dict)), default=0.0)
        stress_field = {}
        if self.backbone and self.backbone.partition:
            stress_field = copy.deepcopy(self.backbone.partition.stress_field)
        return {
            "frame_idx": frame_idx,
            "asset_states": copy.deepcopy(p2_package.get("assets", {})),
            "stress_field": stress_field,
            "crease_field": copy.deepcopy(p2_package.get("crease_field", {})),
            "tear_field": copy.deepcopy(p2_package.get("tear_field", {})),
            "global_tile_info": copy.deepcopy(tile_info),
            "max_stress": max_stress,
            "active_ciphers": p2_package.get("ciphers", []),
            "material_dynamics": self._extract_physics_from_p2(p2_package),
        }

    def _run_physics_evolution(self, start_frame, end_frame, total_frames, ciphers, base_physics,
                               initial_assets, hand_grab_sequence=None, impact_sequence=None):
        if (not isinstance(end_frame, torch.Tensor) or end_frame.numel() == 0 or 
            end_frame.shape[0] == 0 or start_frame.shape != end_frame.shape):
            print("⚠️ [P3 演化拦截] End 帧尺寸不合法，使用 Start 帧镜像对齐。")
            end_frame = start_frame.clone()

        diff = (start_frame - end_frame).abs().mean().item()
        if diff < 1e-4:
            print(f"⚠️ [P3] 首尾帧差异极小 ({diff:.6f})，注入微量随机扰动以激活演化。")
            end_frame = end_frame + torch.randn_like(end_frame) * 0.01
            end_frame = end_frame.clamp(-1.0, 1.0)

        dt = 1.0 / self.fps
        all_snapshots = []

        assets = copy.deepcopy(initial_assets)
        self._apply_cipher_driven_physics(assets, ciphers)

        current_p2 = {
            "assets": assets,
            "p1_shadow_frames": torch.stack([start_frame, end_frame], dim=0),
            "ciphers": ciphers,
            "base_physics": base_physics
        }

        print(f"\n🔎 [P3] 资产数: {len(assets)} | 键: {list(assets.keys())[:5]}...")

        if not self.backbone.stress_points and not self.backbone.collision_points:
            H_phys, W_phys = start_frame.shape[-2:]
            normal = start_frame[0, :3, :, :]
            grad_x = torch.abs(normal[:, :, 1:] - normal[:, :, :-1])
            grad_y = torch.abs(normal[:, 1:, :] - normal[:, :-1, :])
            grad_x = F.pad(grad_x, (0, 1, 0, 0)).mean(dim=0, keepdim=True)
            grad_y = F.pad(grad_y, (0, 0, 0, 1)).mean(dim=0, keepdim=True)
            edge_map = (grad_x + grad_y) / 2.0
            grid_h, grid_w = min(32, H_phys), min(32, W_phys)
            downsampled_edge = F.interpolate(edge_map.unsqueeze(0), size=(grid_h, grid_w), mode='area').squeeze(0).squeeze(0)
            threshold = downsampled_edge.quantile(0.90)
            high_stress_y, high_stress_x = torch.where(downsampled_edge > threshold)
            target_keys = set()
            for aid, adata in assets.items():
                if isinstance(adata, dict):
                    target_keys.add(aid)
                    sov = adata.get("sovereignty", "")
                    if sov:
                        target_keys.add(sov)
                        target_keys.add(sov.replace("_factory", ""))
            if not target_keys:
                target_keys.add("skin")
            stress_points = {}
            for y, x in zip(high_stress_y.tolist(), high_stress_x.tolist()):
                nx = x / grid_w
                ny = y / grid_h
                mag = downsampled_edge[y, x].item()
                for k in target_keys:
                    stress_points.setdefault(k, []).append((nx, ny, mag))
            self.backbone.set_stress_points(stress_points)
            self.backbone.set_camera_view(0.5, 0.5, 1.0, 1.0)

        print(f"[P3] 演化启动，目标步数: {total_frames}")
        for frame_idx in range(total_frames):
            hand_grab = hand_grab_sequence[frame_idx] if hand_grab_sequence else None
            impacts = impact_sequence[frame_idx] if impact_sequence else None
            current_p2 = self._process_single_frame(frame_idx, dt, current_p2, hand_grab, impacts)
            if self.backbone:
                current_p2 = self.backbone.process_frame(current_p2, dt)

            if frame_idx % 5 == 0 or frame_idx == total_frames - 1:
                tile_info = current_p2.get("global_tile_info", {})
                stress_vals = [v["stress_mag"] for v in tile_info.values() if isinstance(v, dict) and "stress_mag" in v]
                avg_s = sum(stress_vals)/len(stress_vals) if stress_vals else 0.0
                max_s = max(stress_vals) if stress_vals else 0.0
                print(f"  ⏱ 帧 {frame_idx:02d}/{total_frames-1} | 瓦片: {len(stress_vals)} | 应力: 均值={avg_s:.3f} 峰值={max_s:.3f}")

            snapshot = self._build_snapshot_from_p2(frame_idx, current_p2)
            all_snapshots.append(snapshot)

        self._diagnose_backbone_state("演化完成")
        return all_snapshots

    def _diagnose_backbone_state(self, prefix: str = ""):
        bb = self.backbone
        p = bb.partition
        sf = p.stress_field
        tiles = p.tiles
        sp = bb.stress_points
        print(f"\n{'='*60}")
        print(f"📊 {prefix} 应力场状态速查")
        print(f"  应力点资产组: {len(sp)} | 总点数: {sum(len(v) for v in sp.values())}")
        print(f"  stress_field 瓦片数: {len(sf)} | 应力均值: {sum(v[0] for v in sf.values())/max(1,len(sf)):.4f}" if sf else "  stress_field: 空")
        print(f"  tiles 瓦片数: {len(tiles)} | 存储应力均值: {sum(t.stress_magnitude for t in tiles.values())/max(1,len(tiles)):.4f}" if tiles else "  tiles: 空")
        print(f"{'='*60}\n")

    def _maybe_apply_neural_sync(self, unet_module: Optional[nn.Module] = None):
        pass

    # ---------- 核心物理图构建（智能实体识别版：修复 global_physics 优先读取）----------
    def _build_physics_map_from_snapshot(self, snap: Dict[str, Any], H: int, W: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phys = torch.zeros((48, H, W), dtype=torch.float32, device=device)

        # 0-15 通道：应力场空间分布
        stress_field = snap.get("stress_field", {})
        if stress_field and self.backbone:
            tiles_x = self.backbone.partition.tiles_x
            tiles_y = self.backbone.partition.tiles_y
            scale_x = W / tiles_x
            scale_y = H / tiles_y
            for (tx, ty), (mag, (dx, dy)) in stress_field.items():
                x_start = int(tx * scale_x)
                y_start = int(ty * scale_y)
                x_end = int((tx + 1) * scale_x)
                y_end = int((ty + 1) * scale_y)
                phys[0, y_start:y_end, x_start:x_end] = mag
                phys[1, y_start:y_end, x_start:x_end] = dx
                phys[2, y_start:y_end, x_start:x_end] = dy
            for c in range(3, 16):
                phys[c] = phys[0]

            print(f"[P3 PhysicsMap] 应力通道 (0-15) 统计 (零均值前):")
            print(f"    phys[0] mean={phys[0].mean().item():.6f} std={phys[0].std().item():.6f} min={phys[0].min().item():.6f} max={phys[0].max().item():.6f}")

            # 零均值化
            phys[0:16] = phys[0:16] - phys[0:16].mean(dim=(-2, -1), keepdim=True)

            print(f"    phys[0] (零均值后) mean={phys[0].mean().item():.6f} std={phys[0].std().item():.6f} min={phys[0].min().item():.6f} max={phys[0].max().item():.6f}")

            # 可选：轻微增益，可在观察日志后手动调整这个值
            gain = 1.0
            if gain != 1.0:
                phys[0:16] = phys[0:16] * gain
                print(f"    已对 0-15 通道施加增益: x{gain}")

        else:
            print("[P3 PhysicsMap] 无应力场数据，前 16 通道保持为零")

        # 16-47 通道：全局物理常量（智能实体识别版）
        mat_dyn = snap.get("material_dynamics", {})
        if mat_dyn:
            # 优先使用 global_physics，避免取到 global_ciphers 等元数据
            if "global_physics" in mat_dyn:
                target_entity = mat_dyn["global_physics"]
            else:
                # 过滤掉已知的非物理实体键
                valid_entities = {k: v for k, v in mat_dyn.items() if k not in ("global_ciphers", "manifest")}
                target_entity = next(iter(valid_entities.values()), {}) if valid_entities else {}

            wetness = float(target_entity.get("wetness", 0.0))
            friction = float(target_entity.get("friction", 0.5))
            elasticity = float(target_entity.get("elasticity", 0.5))
            stiffness = float(target_entity.get("stiffness", 0.5))
            tear = float(target_entity.get("tear_intensity", 0.0))
            tension = float(target_entity.get("tension", 0.0))
            body_type = float(target_entity.get("body_type", 0.5))
        else:
            wetness = 0.0; friction = 0.5; elasticity = 0.5; stiffness = 0.5
            tear = 0.0; tension = 0.0; body_type = 0.5

        phys[16] = wetness; phys[17] = friction; phys[18] = elasticity; phys[19] = stiffness
        for c in range(20, 24): phys[c] = wetness
        for c in range(24, 28): phys[c] = friction
        for c in range(28, 32): phys[c] = elasticity
        for c in range(32, 36): phys[c] = stiffness
        for c in range(36, 40): phys[c] = tear
        for c in range(40, 44): phys[c] = tension
        for c in range(44, 48): phys[c] = body_type

        return phys

    # ---------- 生成载荷（完整快照传递）----------
    def _generate_payloads(self, all_snapshots, scale_factor, total_frames, ciphers, base_physics, seed, user_steps, start_frame_for_blend=None):
        all_states = [snap["material_dynamics"] for snap in all_snapshots]
        adaptive_scale = scale_factor * (user_steps / max(total_frames, 1)) if user_steps > 0 else scale_factor

        H, W = self._internal_8x_h, self._internal_8x_w

        physics_maps = []
        for snap in all_snapshots:
            phys_map = self._build_physics_map_from_snapshot(snap, H, W)
            physics_maps.append(phys_map.unsqueeze(0))
        physics_maps = torch.cat(physics_maps, dim=0)

        print(f"[P3] Hijacker 预计算: {total_frames} 帧, 物理图形状: {physics_maps.shape}")
        self.hijacker.clear_all_cache()
        self.hijacker.precompute_sequence_from_states(
            total_frames=total_frames,
            ciphers=ciphers,
            base_params=base_physics,
            all_states=all_states,
            spatial_size=(H, W),
            strength_scale=adaptive_scale,
            seed=seed,
            precomputed_physics=physics_maps
        )

        if hasattr(self.hijacker, '_executor'):
            self.hijacker._executor.shutdown(wait=True)
            import concurrent.futures
            self.hijacker._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        first_payload = self.hijacker.request_payload(0)
        full_48ch = torch.cat([
            first_payload["block_A"],
            first_payload["block_B"],
            first_payload["block_C"]
        ], dim=1)
        self._first_frame_4ch = full_48ch[:, :4, :, :].detach().clone().cpu()
        self._first_frame_metadata = {
            "frame_idx": 0, "ciphers": ciphers,
            "base_params": base_physics, "evolved_params": first_payload["evolved_params"]
        }
        print(f"[P3] 首帧 payload 生成完成，evolved_params: {first_payload['evolved_params']}")
        self._cached_memory_vectors = [self.memory_encoder.encode(snap, ciphers) for snap in all_snapshots]

    def _filter_conditioning(self, positive, negative):
        return positive, negative

    def _assemble_p3_payload(self, fingerprint, total_frames, golden_info, anchor_applied, conflict_log):
        cache_dir = self.hijacker.cache_dir
        frame_file_paths = [os.path.abspath(os.path.join(cache_dir, f"frame_{i:06d}_tri.safetensors")) for i in range(total_frames)]
        diff_params = self.current_p2_package.get("p2_diffusion_params", {}) or {
            "num_inference_steps": self.current_p2_package.get("steps", 20),
            "cfg_scale": self.current_p2_package.get("cfg", 7.5),
            "seed": self.current_p2_package.get("seed", 42),
            "sampler_type": self.current_p2_package.get("sampler_name", "euler"),
            "scheduler": self.current_p2_package.get("scheduler", "normal"),
        }

        gpu_first_frame = self._first_frame_4ch.clone()
        if torch.cuda.is_available():
            gpu_first_frame = gpu_first_frame.to("cuda")

        return {
            "main_controller": self,
            "first_frame_4ch": gpu_first_frame,
            "frame_file_paths": frame_file_paths,
            "frame_metadata_list": [{} for _ in range(total_frames)],
            "golden_suggestions": golden_info,
            "anchor_correction_applied": anchor_applied,
            "conflict_log": conflict_log,
            "fingerprint": fingerprint,
            "latent_raw_info": self.lat_raw_info,
            "dim_sync": {"channels": 48, "frames": total_frames, "layout": "BCFHW", "is_high_dim": True, "num_slices": 1},
            "p2_diffusion_params": diff_params
        }

    @torch.inference_mode()
    def process(self, model, positive, negative, latent, p2_package):
        print("\n[P3:Controller] ======= 核心处理流开始 =======")
        self.current_p2_package = p2_package

        raw_assets = p2_package.get("assets", {})
        print(f"[P3] 接收到的资产数: {len(raw_assets)}, 键: {list(raw_assets.keys())}")

        base_physics = None
        if "global_physics" in raw_assets and isinstance(raw_assets["global_physics"], dict):
            base_physics = raw_assets["global_physics"]["slots"]["slot_6_physics_layer"]
            print(f"[P3] 从 global_physics 实体加载 base_physics: {base_physics}")
        else:
            base_physics = p2_package.get("base_physics", self.GLOBAL_DEFAULT_PHYSICS.copy())
            print(f"[P3] 使用默认/传入的 base_physics")

        start_frame = p2_package.get("start_frame")
        end_frame = p2_package.get("end_frame")
        if start_frame is None or end_frame is None:
            p1_frames = p2_package["p1_shadow_frames"]
            start_frame = p1_frames[0:1]
            end_frame = p1_frames[1:2] if p1_frames.shape[0] >= 2 else p1_frames[0:1]

        all_ciphers = p2_package.get("ciphers", [])
        diff_params = p2_package.get("p2_diffusion_params", {})
        seed = diff_params.get("seed", p2_package.get("seed", 42)) if diff_params else p2_package.get("seed", 42)
        cfg = diff_params.get("cfg_scale", p2_package.get("cfg", 7.5)) if diff_params else p2_package.get("cfg", 7.5)
        user_steps = diff_params.get("num_inference_steps", p2_package.get("steps", 20)) if diff_params else p2_package.get("steps", 20)

        latent_tensor = latent["samples"]
        if latent_tensor.dim() == 5:
            B, C, F, H, W = latent_tensor.shape
        elif latent_tensor.dim() == 4:
            B, C, H, W = latent_tensor.shape; F = 1
        else:
            H, W, B, C, F = 256, 256, 1, 4, 1

        self._internal_8x_h = H
        self._internal_8x_w = W
        self.lat_raw_info.update({"orig_height": H, "orig_width": W, "total_frames": F})
        total_frames = F
        print(f"[P3] 潜空间分辨率: {H}x{W}, 总帧数: {total_frames}")

        initial_assets = p2_package.get("assets", {})
        self._init_modules({"assets": initial_assets}, seed=seed)

        fingerprint = self._compute_fingerprint(start_frame, end_frame, all_ciphers, base_physics)
        fingerprint = f"{fingerprint}_S{seed}_C{cfg}_T{total_frames}"
        print(f"[P3] 指纹: {fingerprint}")

        slot_data = self.slot_manager.query(fingerprint)
        if slot_data:
            all_snapshots = slot_data["all_snapshots"]
            golden_info = slot_data["golden_info"]
            anchor_applied = slot_data["anchor_correction_applied"]
            print("[P3] 🎯 缓存命中，跳过演化")
        else:
            all_snapshots = self._run_physics_evolution(start_frame, end_frame, total_frames, all_ciphers, base_physics, initial_assets)
            corrected_snapshots, anchor_applied, _ = self.anchor_corrector.check_and_correct(all_snapshots, end_frame)
            all_snapshots = corrected_snapshots
            golden_info = self.golden_evaluator.evaluate(all_snapshots, all_ciphers, 0.1, False, total_frames, user_steps)
            self.slot_manager.store(fingerprint, all_snapshots, golden_info, anchor_applied)

        self._generate_payloads(all_snapshots, golden_info.get("scale_factor", 1.0), total_frames, all_ciphers, base_physics, seed, user_steps, start_frame_for_blend=start_frame)

        if self.enable_neural_sync:
            unet = getattr(model, 'model', None)
            if unet is not None:
                self._maybe_apply_neural_sync(unet)

        positive_out, negative_out = self._filter_conditioning(positive, negative)
        p3_payload = self._assemble_p3_payload(fingerprint, total_frames, golden_info, anchor_applied, [])
        self.current_p3_payload = p3_payload

        print("[P3:Controller] ======= 核心处理流结束 =======\n")
        return model, positive_out, negative_out, latent, p3_payload

    def get_frame_slice(self, frame_idx, slice_idx=0, target_ch=48, device="cuda"):
        payload = self.hijacker.request_payload(frame_idx)
        full_48ch = torch.cat([
            payload["block_A"].to(device),
            payload["block_B"].to(device),
            payload["block_C"].to(device)
        ], dim=1)
        return full_48ch

    def sync_garbage_collection(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def emergency_clean(self):
        if self.hijacker:
            self.hijacker.clear_all_cache()
        self._first_frame_4ch = None
        self._first_frame_metadata = None
        self.sync_garbage_collection()

    def set_dressing_action(self, action):
        self.dressing_action = action


P3MainController = P3Controller