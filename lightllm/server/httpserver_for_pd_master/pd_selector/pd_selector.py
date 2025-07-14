from typing import Union, List, Tuple
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams


class PDSelector:
    def __init__(self, prefill_nodes, decode_nodes):
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes

    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        raise NotImplementedError("Subclass must implement this method")


class RandomSelector(PDSelector):
    """随机选择器"""

    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        import random

        p_node = random.choice(self.prefill_nodes)
        d_node = random.choice(self.decode_nodes)
        return p_node, d_node


class RoundRobinSelector(PDSelector):
    """轮询选择器"""

    def __init__(self, prefill_nodes, decode_nodes):
        super().__init__(prefill_nodes, decode_nodes)
        self.prefill_node_index = 0
        self.decode_node_index = 0

    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        p_node = self.prefill_nodes[self.prefill_node_index]
        d_node = self.decode_nodes[self.decode_node_index]
        self.prefill_node_index = (self.prefill_node_index + 1) % len(self.prefill_nodes)
        self.decode_node_index = (self.decode_node_index + 1) % len(self.decode_nodes)
        return p_node, d_node


class MemorySelector(PDSelector):
    """基于内存使用情况的选择器"""

    def __init__(self, prefill_nodes, decode_nodes):
        super().__init__(prefill_nodes, decode_nodes)
        # 内存使用情况缓存
        self.node_memory_cache = {}  # 节点IP:PORT -> 内存使用率
        self.memory_monitor_task = None
        self._start_memory_monitor()

    def _start_memory_monitor(self):
        """启动内存监控后台任务"""
        import asyncio

        if self.memory_monitor_task is None or self.memory_monitor_task.done():
            try:
                loop = asyncio.get_event_loop()
                self.memory_monitor_task = loop.create_task(self._memory_monitor_loop())
                from lightllm.utils.log_utils import init_logger
                logger = init_logger(__name__)
                logger.info("Started memory monitoring task")
            except RuntimeError:
                from lightllm.utils.log_utils import init_logger
                logger = init_logger(__name__)
                logger.warning("No event loop running, memory monitoring will start later")

    async def _memory_monitor_loop(self):
        """后台内存监控循环，每秒更新一次所有节点的内存使用情况"""
        import aiohttp
        import asyncio
        from lightllm.utils.log_utils import init_logger
        logger = init_logger(__name__)

        async def get_node_memory_usage(node: PD_Client_Obj) -> tuple:
            node_url = f"http://{node.client_ip_port}/token_load"
            timeout = aiohttp.ClientTimeout(total=3.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(node_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        current_load = data.get("current_load", 0.0)
                        if isinstance(current_load, list):
                            current_load = current_load[0] if current_load else 0.0
                        return node.client_ip_port, float(current_load)
                    else:
                        logger.warning(f"Failed to get token_load from {node.client_ip_port}, status: {response.status}")
                        return node.client_ip_port, float('inf')

        while True:
            all_nodes = self.prefill_nodes + self.decode_nodes
            tasks = [get_node_memory_usage(node) for node in all_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 更新缓存
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    node_key, usage = result
                    self.node_memory_cache[node_key] = usage

            logger.debug(f"Updated memory cache: {self.node_memory_cache}")

            # 等待1秒后再次检查
            await asyncio.sleep(1.0)

    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        # 获取prefill节点的内存使用情况
        prefill_usages = [self.node_memory_cache.get(node.client_ip_port, float('inf')) for node in self.prefill_nodes]
        decode_usages = [self.node_memory_cache.get(node.client_ip_port, float('inf')) for node in self.decode_nodes]

        import random
        min_prefill_usage = min(prefill_usages)
        min_decode_usage = min(decode_usages)
        p_node = self.prefill_nodes[prefill_usages.index(min_prefill_usage)] if min_prefill_usage != float('inf') else random.choice(self.prefill_nodes)
        d_node = self.decode_nodes[decode_usages.index(min_decode_usage)] if min_decode_usage != float('inf') else random.choice(self.decode_nodes)

        return p_node, d_node

    def remove_node_cache(self, node_ip_port: str):
        """删除节点的内存缓存"""
        self.node_memory_cache.pop(node_ip_port, None)


class RadixSelector(PDSelector):
    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        pass