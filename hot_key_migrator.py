"""
分布式一致性哈希环上的热键迁移调度器
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import mmh3

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
DEFAULT_VNODE_COUNT = 150
VNODE_ADJUST_STEP = 5
VNODE_MIN = 30
VNODE_MAX = 300
HOT_KEY_THRESHOLD = 1000
WINDOW_SIZE = 60  # 秒


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────
@dataclass
class MigrationStep:
    from_node: str
    to_node: str
    keys_affected: List[str]
    vnode_delta: int  # from_node 减少 / to_node 增加的虚拟节点数


# ──────────────────────────────────────────────
# 一致性哈希环
# ──────────────────────────────────────────────
class ConsistentHashRing:
    def __init__(self, vnode_count: int = DEFAULT_VNODE_COUNT):
        self._vnode_count = vnode_count  # 每个节点的默认虚拟节点数
        # node_id -> 当前虚拟节点数
        self._node_vnode_count: Dict[str, int] = {}
        # 排序的虚拟节点哈希列表
        self._sorted_hashes: List[int] = []
        # 哈希 -> node_id
        self._hash_to_node: Dict[int, str] = {}
        # node_id -> 该节点所有虚拟节点哈希集合
        self._node_hashes: Dict[str, Set[int]] = {}

    @staticmethod
    def _hash(key: str) -> int:
        return mmh3.hash(key, seed=42) & 0xFFFFFFFF

    def _vnode_key(self, node_id: str, index: int) -> str:
        return f"{node_id}#vn{index}"

    def _add_vnodes_for_node(self, node_id: str, count: int) -> None:
        """为节点添加 count 个虚拟节点"""
        existing = self._node_hashes.get(node_id, set())
        start_idx = len(existing)
        for i in range(count):
            vnode_key = self._vnode_key(node_id, start_idx + i)
            h = self._hash(vnode_key)
            if h in self._hash_to_node:
                continue  # 哈希冲突，跳过（极低概率）
            self._hash_to_node[h] = node_id
            bisect.insort(self._sorted_hashes, h)
            existing.add(h)
        self._node_hashes[node_id] = existing

    def _remove_vnodes_for_node(self, node_id: str, count: int) -> None:
        """从节点移除 count 个虚拟节点（移除哈希值最大的那些）"""
        hashes = self._node_hashes.get(node_id, set())
        # 按哈希值降序排列，移除最后添加的
        sorted_hashes = sorted(hashes, reverse=True)
        removed = 0
        for h in sorted_hashes:
            if removed >= count:
                break
            idx = bisect.bisect_left(self._sorted_hashes, h)
            if idx < len(self._sorted_hashes) and self._sorted_hashes[idx] == h:
                self._sorted_hashes.pop(idx)
            self._hash_to_node.pop(h, None)
            hashes.discard(h)
            removed += 1
        self._node_hashes[node_id] = hashes

    def add_node(self, node_id: str, vnode_count: Optional[int] = None) -> None:
        """添加节点"""
        if node_id in self._node_vnode_count:
            return
        count = vnode_count if vnode_count is not None else self._vnode_count
        self._node_vnode_count[node_id] = count
        self._add_vnodes_for_node(node_id, count)

    def remove_node(self, node_id: str) -> None:
        """移除节点"""
        if node_id not in self._node_vnode_count:
            return
        self._remove_vnodes_for_node(node_id, self._node_vnode_count[node_id])
        del self._node_vnode_count[node_id]
        self._node_hashes.pop(node_id, None)

    def adjust_vnodes(self, node_id: str, delta: int) -> bool:
        """调整节点的虚拟节点数量，delta > 0 增加，delta < 0 减少。
        返回是否调整成功。"""
        if node_id not in self._node_vnode_count:
            return False
        new_count = self._node_vnode_count[node_id] + delta
        if new_count < VNODE_MIN or new_count > VNODE_MAX:
            return False
        if delta > 0:
            self._add_vnodes_for_node(node_id, delta)
        elif delta < 0:
            self._remove_vnodes_for_node(node_id, -delta)
        self._node_vnode_count[node_id] = new_count
        return True

    def get_node(self, key: str) -> Optional[str]:
        """返回 key 映射到的节点，环为空返回 None"""
        if not self._sorted_hashes:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_hashes, h)
        if idx == len(self._sorted_hashes):
            idx = 0
        return self._hash_to_node[self._sorted_hashes[idx]]

    def get_all_node_ids(self) -> List[str]:
        return list(self._node_vnode_count.keys())

    def get_vnode_count(self, node_id: str) -> int:
        return self._node_vnode_count.get(node_id, 0)

    def get_key_mapping(self, keys: List[str]) -> Dict[str, str]:
        """批量获取 key -> node 映射"""
        return {k: self.get_node(k) for k in keys}

    @property
    def node_count(self) -> int:
        return len(self._node_vnode_count)


# ──────────────────────────────────────────────
# 滑动窗口热键检测
# ──────────────────────────────────────────────
class SlidingWindowCounter:
    """基于秒级精度的滑动窗口计数器，线程安全，细粒度锁"""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        # key -> {second -> count}
        self._counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # 细粒度锁：每个 key 一把锁
        self._key_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        # 全局快照锁（读写锁模拟）
        self._snapshot_lock = threading.Lock()

    def record_access(self, key: str, timestamp: float) -> None:
        """记录一次访问，timestamp 为 Unix 时间戳（秒）"""
        sec = int(timestamp)
        lock = self._key_locks[key]
        with lock:
            self._counts[key][sec] += 1

    def get_count(self, key: str, now: float) -> int:
        """获取 key 在滑动窗口内的访问计数"""
        sec_now = int(now)
        total = 0
        with self._snapshot_lock:
            key_data = self._counts.get(key)
            if key_data is None:
                return 0
            # 复制一份避免长时间持锁
            items = list(key_data.items())
        for s, c in items:
            if sec_now - s < self._window_size:
                total += c
        return total

    def get_hot_keys(self, now: float, threshold: int = HOT_KEY_THRESHOLD) -> Dict[str, int]:
        """返回所有热键及其访问计数"""
        sec_now = int(now)
        result: Dict[str, int] = {}
        with self._snapshot_lock:
            # 快照：复制所有 key 的数据
            snapshot = {k: dict(v) for k, v in self._counts.items()}
        for key, sec_counts in snapshot.items():
            total = sum(c for s, c in sec_counts.items() if sec_now - s < self._window_size)
            if total >= threshold:
                result[key] = total
        return result

    def get_all_key_counts(self, now: float) -> Dict[str, int]:
        """返回所有 key 的访问计数"""
        sec_now = int(now)
        result: Dict[str, int] = {}
        with self._snapshot_lock:
            snapshot = {k: dict(v) for k, v in self._counts.items()}
        for key, sec_counts in snapshot.items():
            total = sum(c for s, c in sec_counts.items() if sec_now - s < self._window_size)
            result[key] = total
        return result

    def cleanup(self, now: float) -> None:
        """清理过期数据"""
        sec_now = int(now)
        with self._snapshot_lock:
            expired_keys = []
            for key, sec_counts in self._counts.items():
                expired_secs = [s for s in sec_counts if sec_now - s >= self._window_size]
                for s in expired_secs:
                    del sec_counts[s]
                if not sec_counts:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._counts[key]


# ──────────────────────────────────────────────
# 热键迁移调度器
# ──────────────────────────────────────────────
class HotKeyMigrator:
    def __init__(self, vnode_count: int = DEFAULT_VNODE_COUNT):
        self.ring = ConsistentHashRing(vnode_count)
        self.counter = SlidingWindowCounter()
        self._all_keys: Set[str] = set()
        self._keys_lock = threading.Lock()

    def add_node(self, node_id: str) -> None:
        self.ring.add_node(node_id)

    def remove_node(self, node_id: str) -> None:
        self.ring.remove_node(node_id)

    def record_access(self, key: str, timestamp: Optional[float] = None) -> None:
        """记录 key 访问，同时追踪 key 集合"""
        if timestamp is None:
            timestamp = time.time()
        with self._keys_lock:
            self._all_keys.add(key)
        self.counter.record_access(key, timestamp)

    def get_node(self, key: str) -> Optional[str]:
        return self.ring.get_node(key)

    def get_hot_keys(self, now: Optional[float] = None) -> Dict[str, int]:
        if now is None:
            now = time.time()
        return self.counter.get_hot_keys(now)

    def _compute_node_load(self, now: float) -> Dict[str, float]:
        """计算每个节点的负载（基于访问计数）"""
        key_counts = self.counter.get_all_key_counts(now)
        node_load: Dict[str, float] = defaultdict(float)
        for key, count in key_counts.items():
            node = self.ring.get_node(key)
            if node:
                node_load[node] += count
        # 确保所有节点都有负载值
        for node_id in self.ring.get_all_node_ids():
            if node_id not in node_load:
                node_load[node_id] = 0.0
        return node_load

    def compute_migration_plan(
        self,
        threshold: float,
        now: Optional[float] = None,
    ) -> List[MigrationStep]:
        """
        计算迁移计划。
        threshold: 负载阈值，超过此值的节点为过载节点。
        迁移策略：通过调整虚拟节点数量，将热键从过载节点迁移到低负载节点。
        每次调整 VNODE_ADJUST_STEP 个虚拟节点。
        """
        if now is None:
            now = time.time()

        node_load = self._compute_node_load(now)
        all_keys = list(self._all_keys)

        # 获取当前 key -> node 映射
        key_to_node = self.ring.get_key_mapping(all_keys)

        # 分类节点
        overloaded = {n for n, load in node_load.items() if load > threshold}
        underloaded = {n for n, load in node_load.items() if load < threshold * 0.5}

        if not overloaded or not underloaded:
            return []

        # 对过载节点按负载降序排列
        overloaded_sorted = sorted(overloaded, key=lambda n: node_load[n], reverse=True)
        # 对低负载节点按负载升序排列（最空闲的优先接收）
        underloaded_sorted = sorted(underloaded, key=lambda n: node_load[n])

        plan: List[MigrationStep] = []

        for from_node in overloaded_sorted:
            # 找出该节点上的所有 key
            node_keys = [k for k, n in key_to_node.items() if n == from_node]
            # 按访问量降序排列
            key_counts = self.counter.get_all_key_counts(now)
            node_keys.sort(key=lambda k: key_counts.get(k, 0), reverse=True)

            for to_node in underloaded_sorted:
                # 检查虚拟节点数量约束
                from_vnodes = self.ring.get_vnode_count(from_node)
                to_vnodes = self.ring.get_vnode_count(to_node)
                if from_vnodes - VNODE_ADJUST_STEP < VNODE_MIN:
                    continue
                if to_vnodes + VNODE_ADJUST_STEP > VNODE_MAX:
                    continue

                # 临时调整虚拟节点，计算受影响的 key
                # 保存当前映射
                old_mapping = dict(key_to_node)

                # 临时调整
                self.ring.adjust_vnodes(from_node, -VNODE_ADJUST_STEP)
                self.ring.adjust_vnodes(to_node, VNODE_ADJUST_STEP)

                # 计算新的映射
                new_mapping = self.ring.get_key_mapping(all_keys)

                # 找出受影响的 key（从 from_node 迁出的）
                affected_keys = [
                    k for k in node_keys
                    if old_mapping.get(k) == from_node and new_mapping.get(k) != from_node
                ]

                # 回滚调整
                self.ring.adjust_vnodes(from_node, VNODE_ADJUST_STEP)
                self.ring.adjust_vnodes(to_node, -VNODE_ADJUST_STEP)

                if affected_keys:
                    plan.append(MigrationStep(
                        from_node=from_node,
                        to_node=to_node,
                        keys_affected=affected_keys,
                        vnode_delta=VNODE_ADJUST_STEP,
                    ))

                    # 更新负载估算
                    migrated_load = sum(key_counts.get(k, 0) for k in affected_keys)
                    node_load[from_node] -= migrated_load
                    node_load[to_node] += migrated_load

                    # 如果 from_node 不再过载，跳出
                    if node_load[from_node] <= threshold:
                        break

            # 如果 from_node 仍然过载但已经没有可用的 to_node，继续下一个 from_node

        return plan

    def execute_migration_plan(self, plan: List[MigrationStep]) -> None:
        """执行迁移计划（调整虚拟节点数量）"""
        for step in plan:
            self.ring.adjust_vnodes(step.from_node, -step.vnode_delta)
            self.ring.adjust_vnodes(step.to_node, step.vnode_delta)

    def get_node_load_variance(self, now: Optional[float] = None) -> float:
        """计算节点负载方差"""
        if now is None:
            now = time.time()
        node_load = self._compute_node_load(now)
        if not node_load:
            return 0.0
        values = list(node_load.values())
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
