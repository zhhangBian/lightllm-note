from typing import Union, List, Tuple
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams


class PDSelector:
    def __init__(self, prefill_nodes: List[PD_Client_Obj], decode_nodes: List[PD_Client_Obj], pd_manager):
        self.prefill_nodes: List[PD_Client_Obj] = prefill_nodes
        self.decode_nodes: List[PD_Client_Obj] = decode_nodes
        self.pd_manager = pd_manager

    async def update_nodes(self, prefill_nodes, decode_nodes):
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

    def __init__(self, prefill_nodes: List[PD_Client_Obj], decode_nodes: List[PD_Client_Obj], pd_manager):
        super().__init__(prefill_nodes, decode_nodes, pd_manager)
        self.prefill_node_index: int = 0
        self.decode_node_index: int = 0

    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        p_node = self.prefill_nodes[self.prefill_node_index]
        d_node = self.decode_nodes[self.decode_node_index]
        self.prefill_node_index = (self.prefill_node_index + 1) % len(self.prefill_nodes)
        self.decode_node_index = (self.decode_node_index + 1) % len(self.decode_nodes)
        return p_node, d_node


class MemorySelector(PDSelector):
    """基于内存使用情况的选择器"""

    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        def _get_min_node(node_infos: dict, key: str):
            min_node, min_node_len = None, float("inf")
            for node_ip, node_info in node_infos.items():
                if node_info[key] < float("inf"):
                    if node_info[key] < min_node_len:
                        min_node_len = node_info[key]
                        min_node = node_ip
            return min_node

        if self.pd_manager is None:
            # 如果没有 PDManager 引用，回退到随机选择
            import random
            p_node = random.choice(self.prefill_nodes) if self.prefill_nodes else None
            d_node = random.choice(self.decode_nodes) if self.decode_nodes else None
            return p_node, d_node

        node_infos = self.pd_manager.get_predict_node_infos()
        node_infos = {k: v for k, v in node_infos.items() if v["mem_len"] < float("inf")}
        if len(node_infos) == 0:
            return random.choice(self.prefill_nodes), random.choice(self.decode_nodes)

        # 获取负载最小的节点
        p_node_infos = {k: v for k, v in node_infos.items() if k in self.prefill_nodes}
        d_node_infos = {k: v for k, v in node_infos.items() if k in self.decode_nodes}
        p_node = _get_min_node(p_node_infos, "mem_len") or random.choice(self.prefill_nodes)
        d_node = _get_min_node(d_node_infos, "mem_len") or random.choice(self.decode_nodes)

        return p_node, d_node

class RadixSelector(PDSelector):
    async def select_p_d_node(self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        pass
