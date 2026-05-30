# p4_prefetch_scheduler.py (全精度跨流同步与状态机完美版)
import os
import time
import threading
import torch
from collections import deque
from typing import Optional, Dict, Any, Tuple
from safetensors.torch import load_file

class P4PrefetchScheduler:
    def __init__(self, frame_file_paths: list, device: torch.device, prefetch_depth: int = 2):
        self.frame_paths = frame_file_paths
        self.device = device
        self.total_frames = len(frame_file_paths)
        self.prefetch_depth = min(prefetch_depth, self.total_frames - 1)

        # 队列存储三元组: (frame_idx, data_gpu, cuda_event)
        self._prefetch_queue = deque(maxlen=self.prefetch_depth + 1)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        self._last_consumed_idx = -1
        self._stop_event = threading.Event()
        
        # 状态机：None=未处理, 'fetching'=正在下载/拷贝, 'ready'=已在队列中
        self._ready_flags: Dict[int, str] = {}

        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def _get_next_to_prefetch(self) -> int:
        with self._lock:
            start_idx = self._last_consumed_idx + 1
            # 在预取视窗内，寻找第一个未被捕获（不在状态机里）的帧
            for offset in range(self.prefetch_depth + 1):
                candidate = start_idx + offset
                if candidate >= self.total_frames:
                    break
                if candidate not in self._ready_flags:
                    return candidate
            return -1

    def _check_vram(self) -> bool:
        if torch.cuda.is_available():
            # 💡 放弃硬编码。动态获取当前可用显存
            free_mem, _ = torch.cuda.mem_get_info(self.device)
            # 如果整张卡剩余可用显存低于 1.5 GB，安全暂停预取，防止挤爆 OOM
            if free_mem < 1.5 * 1024**3:
                return False
        return True

    def _prefetch_worker(self):
        while not self._stop_event.is_set():
            if not self._check_vram():
                time.sleep(0.05)
                continue

            with self._condition:
                if len(self._prefetch_queue) >= self.prefetch_depth + 1:
                    self._condition.wait(timeout=0.01)
                    continue

                target_idx = self._get_next_to_prefetch()
                if target_idx == -1:
                    self._condition.wait(timeout=0.01)
                    continue

                # 标记该帧正在处理，锁住状态机
                with self._lock:
                    self._ready_flags[target_idx] = 'fetching'

                path = self.frame_paths[target_idx]
                if not os.path.exists(path):
                    with self._lock:
                        self._ready_flags.pop(target_idx, None)
                    continue

                try:
                    data_cpu = load_file(path, device="cpu")
                    pinned = {k: v.pin_memory() if v.is_cpu else v for k, v in data_cpu.items()}
                    
                    # 子线程异步推向 GPU
                    data_gpu = {k: v.float().to(self.device, non_blocking=True) for k, v in pinned.items()}
                    
                    # 🛡️ 核心修复：建立硬件级跨流事件同步锁
                    event = None
                    if torch.cuda.is_available():
                        event = torch.cuda.Event()
                        event.record()  # 记录当前预取流的拷贝节点

                    self._prefetch_queue.append((target_idx, data_gpu, event))
                    
                    with self._lock:
                        self._ready_flags[target_idx] = 'ready'
                        
                except Exception as e:
                    print(f"[P4Prefetch] 预取帧 {target_idx} 失败: {e}")
                    with self._lock:
                        self._ready_flags.pop(target_idx, None)
                finally:
                    self._condition.notify_all()

            time.sleep(0.002)

    def get_frame_data(self, frame_idx: int, blocking: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        with self._condition:
            # 1. 发现跳帧或外部行为不一致，立刻清空队列，重置状态机
            if len(self._prefetch_queue) > 0:
                peek_idx, _, _ = self._prefetch_queue[0]
                if peek_idx != frame_idx:
                    self._prefetch_queue.clear()
                    with self._lock:
                        # 仅保留当前处于 fetching 状态的帧，清除已 ready 的错误标记
                        self._ready_flags = {k: v for k, v in self._ready_flags.items() if v == 'fetching'}

            # 2. 尝试从预取队列命中
            for i, (idx, data, event) in enumerate(list(self._prefetch_queue)):
                if idx == frame_idx:
                    # 清理滞后帧
                    for _ in range(i):
                        self._prefetch_queue.popleft()
                    self._prefetch_queue.popleft()
                    
                    self._last_consumed_idx = frame_idx
                    
                    # 🛡️ 核心修复：强制让主线程的当前流等待预取事件完成，消除撕裂花屏
                    if torch.cuda.is_available() and event is not None:
                        torch.cuda.current_stream().wait_event(event)
                    
                    with self._lock:
                        self._ready_flags.pop(frame_idx, None)
                        
                    self._condition.notify_all()
                    return data

            # 3. 降级通道：未命中则主线程同步阻塞加载
            if blocking:
                path = self.frame_paths[frame_idx]
                if not os.path.exists(path):
                    raise FileNotFoundError(f"帧 {frame_idx} 载荷文件不存在: {path}")
                data_cpu = load_file(path, device="cpu")
                pinned = {k: v.pin_memory() if v.is_cpu else v for k, v in data_cpu.items()}
                # 因为在主线程直接执行，后续算子天然在主线程排队，此处 non_blocking 是完全安全的
                data_gpu = {k: v.float().to(self.device, non_blocking=True) for k, v in pinned.items()}
                self._last_consumed_idx = frame_idx
                
                with self._lock:
                    self._ready_flags.pop(frame_idx, None)
                return data_gpu
            else:
                return None

    def request_prefetch(self, frame_idx: int):
        # 保持接口兼容。新架构下工作线程会自动根据 _last_consumed_idx 向后辐射扫描，
        # 触发此函数主要起到主动唤醒条件锁的作用。
        with self._condition:
            self._condition.notify_all()

    def is_ready(self, frame_idx: int) -> bool:
        with self._lock:
            return self._ready_flags.get(frame_idx) == 'ready'

    def shutdown(self):
        self._stop_event.set()
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
        with self._lock:
            self._prefetch_queue.clear()
            self._ready_flags.clear()

    def __del__(self):
        self.shutdown()