# p3_slot_manager.py (重构版：完整快照序列化 + 指纹稳定性 + LRU 正确性)
import hashlib
import json
import os
import shutil
import time
import torch
from collections import OrderedDict
from typing import Dict, List, Any, Optional

class P3SlotManager:
    """
    物理演化缓存管理器（重构版）
    - 使用 torch.save/load 完整保存快照，杜绝字段丢失
    - 指纹计算保证张量连续性，确保缓存命中率
    - LRU 基于真实时间戳，删除逻辑正确
    """
    def __init__(self, **kwargs):
        # 兼容多种传参：cache_dir 或 storage_dir
        self.cache_dir = kwargs.get("cache_dir") or kwargs.get("storage_dir")
        if not self.cache_dir:
            raise ValueError("[P3SlotManager] 初始化错误：缺少 cache_dir 参数")
        
        self.max_slots = kwargs.get("max_slots", 6)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.slots: OrderedDict[str, Dict] = OrderedDict()
        self.manifest_path = os.path.join(self.cache_dir, "slot_manifest.json")
        self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)
                    for fp, meta in data.items():
                        self.slots[fp] = meta
                # 按最后访问时间排序，最近的在末尾
                self.slots = OrderedDict(
                    sorted(self.slots.items(), key=lambda x: x[1].get('last_access', 0))
                )
            except Exception as e:
                print(f"[P3SlotManager] Manifest 读取警告: {e}")

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(dict(self.slots), f, indent=2)

    def _compute_fingerprint(self, start_frame: torch.Tensor, end_frame: torch.Tensor,
                             ciphers: List[str], base_physics: Dict[str, float]) -> str:
        """计算稳定指纹：强制连续化内存布局，避免哈希抖动"""
        # 确保张量连续且位于 CPU
        start_bytes = start_frame.detach().cpu().contiguous().numpy().tobytes()
        end_bytes = end_frame.detach().cpu().contiguous().numpy().tobytes()
        
        start_hash = hashlib.md5(start_bytes).hexdigest()
        end_hash = hashlib.md5(end_bytes).hexdigest()
        ciphers_str = ','.join(sorted(ciphers))
        physics_str = json.dumps(base_physics, sort_keys=True)
        
        raw = f"{start_hash}_{end_hash}_{ciphers_str}_{physics_str}".encode()
        return hashlib.md5(raw).hexdigest()

    def query(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        查询缓存，返回完整快照列表及元数据
        命中时更新最后访问时间，并将其移到 LRU 末尾
        """
        if fingerprint not in self.slots:
            return None
        
        # 更新访问时间戳
        self.slots[fingerprint]['last_access'] = time.time()
        self.slots.move_to_end(fingerprint)
        # 延迟写 manifest 以降低 I/O 频率（可选，这里保留但可优化）
        self._save_manifest()
        
        slot_dir = os.path.join(self.cache_dir, fingerprint)
        snapshots_path = os.path.join(slot_dir, "snapshots.pt")
        
        if not os.path.exists(snapshots_path):
            # 数据损坏，清理无效条目
            del self.slots[fingerprint]
            self._save_manifest()
            return None
        
        # 加载完整快照列表
        all_snapshots = torch.load(snapshots_path, map_location='cpu')
        
        return {
            'all_snapshots': all_snapshots,
            'golden_info': self.slots[fingerprint].get('golden_info', {}),
            'anchor_correction_applied': self.slots[fingerprint].get('anchor_correction_applied', False)
        }

    def store(self, fingerprint: str, all_snapshots: List[Dict[str, Any]],
              golden_info: Dict[str, Any], anchor_correction_applied: bool):
        """
        存储完整快照列表（使用 torch.save 保留所有 Python 对象与张量）
        LRU 淘汰：超出最大容量时移除最久未使用的条目
        """
        # 淘汰最久未使用的条目（OrderedDict 开头是最旧的）
        while len(self.slots) >= self.max_slots:
            oldest_fp, _ = self.slots.popitem(last=False)
            oldest_dir = os.path.join(self.cache_dir, oldest_fp)
            if os.path.exists(oldest_dir):
                shutil.rmtree(oldest_dir, ignore_errors=True)
        
        slot_dir = os.path.join(self.cache_dir, fingerprint)
        os.makedirs(slot_dir, exist_ok=True)
        
        # 使用 torch.save 完整保存快照列表（包含所有嵌套张量与非张量）
        torch.save(all_snapshots, os.path.join(slot_dir, "snapshots.pt"))
        
        # 更新内存记录
        self.slots[fingerprint] = {
            'total_frames': len(all_snapshots),
            'golden_info': golden_info,
            'anchor_correction_applied': anchor_correction_applied,
            'last_access': time.time()
        }
        # 确保最新条目在末尾
        self.slots.move_to_end(fingerprint)
        self._save_manifest()