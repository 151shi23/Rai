"""
验证测试：一致性哈希环 + 热键迁移调度器
"""

import random
import threading
import time
import unittest
from collections import Counter

from hot_key_migrator import (
    ConsistentHashRing,
    HotKeyMigrator,
    MigrationStep,
    SlidingWindowCounter,
    VNODE_ADJUST_STEP,
    VNODE_MAX,
    VNODE_MIN,
)


class TestConsistentHashRing(unittest.TestCase):
    """一致性哈希环基础测试"""

    def test_empty_ring(self):
        ring = ConsistentHashRing()
        self.assertIsNone(ring.get_node("any_key"))

    def test_single_node(self):
        ring = ConsistentHashRing()
        ring.add_node("node_A")
        # 所有 key 都应该映射到唯一节点
        for i in range(1000):
            self.assertEqual(ring.get_node(f"key_{i}"), "node_A")

    def test_distribution(self):
        """多节点下 key 应该大致均匀分布"""
        ring = ConsistentHashRing()
        nodes = [f"node_{i}" for i in range(10)]
        for n in nodes:
            ring.add_node(n)

        counter = Counter()
        for i in range(100_000):
            node = ring.get_node(f"key_{i}")
            counter[node] += 1

        # 每个节点应该分到约 10% 的 key，允许 ±5%
        for node in nodes:
            ratio = counter[node] / 100_000
            self.assertGreater(ratio, 0.05, f"{node} ratio {ratio} too low")
            self.assertLess(ratio, 0.15, f"{node} ratio {ratio} too high")

    def test_add_node_remapping_ratio(self):
        """增加一个节点后，只有约 1/N 的 key 被重新映射"""
        ring = ConsistentHashRing()
        nodes = [f"node_{i}" for i in range(10)]
        for n in nodes:
            ring.add_node(n)

        keys = [f"key_{i}" for i in range(1_000_000)]
        old_mapping = {k: ring.get_node(k) for k in keys}

        # 添加第 11 个节点
        ring.add_node("node_10")

        remapped = sum(1 for k in keys if ring.get_node(k) != old_mapping[k])
        ratio = remapped / len(keys)
        # 理论上约 1/11 ≈ 9%，允许 3%~15%
        self.assertGreater(ratio, 0.03, f"Remapping ratio {ratio:.4f} too low")
        self.assertLess(ratio, 0.15, f"Remapping ratio {ratio:.4f} too high")

    def test_remove_node_remapping_target(self):
        """删除一个节点后，被重新映射的 key 全部落到顺时针下一节点"""
        ring = ConsistentHashRing()
        nodes = [f"node_{i}" for i in range(10)]
        for n in nodes:
            ring.add_node(n)

        keys = [f"key_{i}" for i in range(100_000)]
        old_mapping = {k: ring.get_node(k) for k in keys}

        # 找出 node_5 上的 key
        node5_keys = [k for k in keys if old_mapping[k] == "node_5"]

        ring.remove_node("node_5")

        # node_5 上的 key 应该全部重新映射
        for k in node5_keys:
            new_node = ring.get_node(k)
            self.assertIsNotNone(new_node)
            self.assertNotEqual(new_node, "node_5")

    def test_remove_node_no_other_remapping(self):
        """删除一个节点后，不在该节点上的 key 不应被重新映射"""
        ring = ConsistentHashRing()
        nodes = [f"node_{i}" for i in range(10)]
        for n in nodes:
            ring.add_node(n)

        keys = [f"key_{i}" for i in range(100_000)]
        old_mapping = {k: ring.get_node(k) for k in keys}

        ring.remove_node("node_5")

        # 不在 node_5 上的 key 不应被重新映射
        for k in keys:
            if old_mapping[k] != "node_5":
                self.assertEqual(ring.get_node(k), old_mapping[k],
                                 f"Key {k} was on {old_mapping[k]}, shouldn't remap")

    def test_adjust_vnodes_constraints(self):
        """虚拟节点数量约束测试"""
        ring = ConsistentHashRing()
        ring.add_node("node_A", vnode_count=VNODE_MIN)

        # 不能低于 VNODE_MIN
        self.assertFalse(ring.adjust_vnodes("node_A", -VNODE_ADJUST_STEP))
        # 可以增加
        self.assertTrue(ring.adjust_vnodes("node_A", VNODE_ADJUST_STEP))
        self.assertEqual(ring.get_vnode_count("node_A"), VNODE_MIN + VNODE_ADJUST_STEP)

    def test_adjust_vnodes_max_constraint(self):
        ring = ConsistentHashRing()
        ring.add_node("node_A", vnode_count=VNODE_MAX)
        # 不能超过 VNODE_MAX
        self.assertFalse(ring.adjust_vnodes("node_A", VNODE_ADJUST_STEP))


class TestSlidingWindowCounter(unittest.TestCase):
    """滑动窗口计数器测试"""

    def test_basic_counting(self):
        counter = SlidingWindowCounter(window_size=60)
        now = time.time()
        for _ in range(100):
            counter.record_access("key1", now)
        self.assertEqual(counter.get_count("key1", now), 100)

    def test_window_expiry(self):
        counter = SlidingWindowCounter(window_size=2)
        now = time.time()
        counter.record_access("key1", now)
        # 3 秒后应该过期
        self.assertEqual(counter.get_count("key1", now + 3), 0)

    def test_hot_keys(self):
        counter = SlidingWindowCounter(window_size=60)
        now = time.time()
        # key1 是热键
        for _ in range(1000):
            counter.record_access("key1", now)
        # key2 不是热键
        for _ in range(500):
            counter.record_access("key2", now)

        hot = counter.get_hot_keys(now, threshold=1000)
        self.assertIn("key1", hot)
        self.assertNotIn("key2", hot)

    def test_concurrent_access(self):
        """并发记录访问"""
        counter = SlidingWindowCounter(window_size=60)
        now = time.time()
        errors = []

        def writer(key, count):
            try:
                for _ in range(count):
                    counter.record_access(key, now)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(f"key_{i}", 1000))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        for i in range(10):
            self.assertEqual(counter.get_count(f"key_{i}", now), 1000)


class TestHotKeyMigrator(unittest.TestCase):
    """热键迁移调度器集成测试"""

    def _setup_migrator(self, num_nodes=10, num_keys=100_000, hot_keys_ratio=0.01):
        """创建测试用的迁移器，热键集中在特定节点上"""
        migrator = HotKeyMigrator()
        for i in range(num_nodes):
            migrator.add_node(f"node_{i}")

        now = time.time()
        all_keys = [f"key_{i}" for i in range(num_keys)]

        # 普通访问
        for key in all_keys:
            migrator.record_access(key, now)

        # 找出映射到 node_0 的 key，将它们变成热键
        node0_keys = [k for k in all_keys if migrator.get_node(k) == "node_0"]
        hot_count = min(int(num_keys * hot_keys_ratio), len(node0_keys))
        hot_keys = node0_keys[:hot_count]
        for key in hot_keys:
            for _ in range(2000):
                migrator.record_access(key, now)

        return migrator, now, all_keys

    def test_migration_plan_creation(self):
        """迁移计划应该被正确创建"""
        migrator, now, _ = self._setup_migrator()
        node_load = migrator._compute_node_load(now)
        # 使用平均负载作为阈值，node_0 远超均值
        threshold = sum(node_load.values()) / len(node_load)

        plan = migrator.compute_migration_plan(threshold, now)
        # 有热键存在，应该有迁移计划
        self.assertGreater(len(plan), 0)

        for step in plan:
            self.assertGreater(len(step.keys_affected), 0)
            self.assertEqual(step.vnode_delta, VNODE_ADJUST_STEP)

    def test_variance_decreases_after_migration(self):
        """迁移后方差应该下降"""
        migrator, now, _ = self._setup_migrator(num_keys=50_000)
        node_load = migrator._compute_node_load(now)
        threshold = sum(node_load.values()) / len(node_load)

        variance_before = migrator.get_node_load_variance(now)

        plan = migrator.compute_migration_plan(threshold, now)
        if plan:
            migrator.execute_migration_plan(plan)
            variance_after = migrator.get_node_load_variance(now)
            self.assertLess(variance_after, variance_before,
                            f"Variance should decrease: {variance_before} -> {variance_after}")

    def test_vnode_constraints_after_migration(self):
        """迁移后虚拟节点数量应在约束范围内"""
        migrator, now, _ = self._setup_migrator()
        node_load = migrator._compute_node_load(now)
        threshold = sum(node_load.values()) / len(node_load)

        plan = migrator.compute_migration_plan(threshold, now)
        migrator.execute_migration_plan(plan)

        for node_id in migrator.ring.get_all_node_ids():
            vc = migrator.ring.get_vnode_count(node_id)
            self.assertGreaterEqual(vc, VNODE_MIN, f"{node_id} vnodes {vc} < {VNODE_MIN}")
            self.assertLessEqual(vc, VNODE_MAX, f"{node_id} vnodes {vc} > {VNODE_MAX}")

    def test_concurrent_record_and_plan(self):
        """并发记录访问和计算迁移计划不应死锁或崩溃"""
        migrator = HotKeyMigrator()
        for i in range(5):
            migrator.add_node(f"node_{i}")

        now = time.time()
        stop_event = threading.Event()
        errors = []

        def writer():
            while not stop_event.is_set():
                try:
                    key = f"key_{random.randint(0, 999)}"
                    migrator.record_access(key, now)
                except Exception as e:
                    errors.append(e)

        def planner():
            for _ in range(5):
                try:
                    migrator.compute_migration_plan(100000, now)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads.append(threading.Thread(target=planner))

        for t in threads:
            t.start()

        time.sleep(1)
        stop_event.set()

        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [], f"Errors during concurrent access: {errors}")

    def test_no_migration_when_balanced(self):
        """负载均衡时不应产生迁移计划"""
        migrator = HotKeyMigrator()
        for i in range(5):
            migrator.add_node(f"node_{i}")

        now = time.time()
        # 每个节点均匀访问
        for i in range(5000):
            key = f"key_{i}"
            migrator.record_access(key, now)
            node = migrator.get_node(key)

        node_load = migrator._compute_node_load(now)
        threshold = sum(node_load.values()) / len(node_load) * 3  # 很高的阈值

        plan = migrator.compute_migration_plan(threshold, now)
        # 负载都在阈值以下，不应有迁移
        self.assertEqual(len(plan), 0)


if __name__ == "__main__":
    unittest.main()
