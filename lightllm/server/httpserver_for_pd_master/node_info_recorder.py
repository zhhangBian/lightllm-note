import copy

from ..pd_io_struct import PD_Client_Obj
from lightllm.server.httpserver.manager import SamplingParams, MultimodalParams
from typing import Union, List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class NodeInfoRecorder:
    def __init__(self):
        self.node_info = {}

    def register_node(self, pd_client: PD_Client_Obj):
        self.node_info[pd_client.client_ip_port] = {
            "node_id": pd_client.node_id,
            "client_ip_port": pd_client.client_ip_port,
            "mode": pd_client.mode,
            "node": pd_client,
            "mem_len": 0,
            # "batch_size": 0,
        }

    def remove_node(self, pd_client: PD_Client_Obj):
        del self.node_info[pd_client.client_ip_port]

    def update_node_load_info(self, load_info: dict):
        if "client_ip_port" in load_info:
            ip_port = load_info["client_ip_port"]
            if ip_port in self.node_info:
                mem_len = load_info["mem_len"]
                # batch_size = load_info["batch_size"]
                self.node_info[ip_port]["mem_len"] = mem_len
                # self.node_info[ip_port]["batch_size"] = batch_size
                logger.debug(f"Updated node load info for {ip_port}: {mem_len}")
                # logger.debug(f"Updated node load info for {ip_port}: {mem_len}, {batch_size}")
            else:
                logger.warning(f"Received load info for unknown node: {ip_port}")
        else:
            logger.warning(f"Received load info without client_ip_port")

    def get_node_infos(self):
        return {k: {
            "mem_len": v.get("mem_len", int("inf")),
            # "batch_size": v.get("batch_size", float("inf")),
        } for k, v in self.node_info.items()}

    def get_node_info(self, client_ip_port: str):
        return self.node_info.get(client_ip_port, None)

class PredictNodeInfoRecorder(NodeInfoRecorder):
    def __init__(self):
        super().__init__()
        self.predict_node_info = {}

    def register_node(self, pd_client: PD_Client_Obj):
        super().register_node(pd_client)
        self.predict_node_info[pd_client.client_ip_port] = copy.copy(self.node_info[pd_client.client_ip_port])

    def remove_node(self, pd_client: PD_Client_Obj):
        super().remove_node(pd_client)
        del self.predict_node_info[pd_client.client_ip_port]

    def update_node_load_info(self, load_info: dict):
        super().update_node_load_info(load_info)
        self.predict_node_info[load_info["client_ip_port"]] = copy.copy(self.node_info[load_info["client_ip_port"]])

    def update_predict_node_info(self, p_node: PD_Client_Obj, d_node: PD_Client_Obj, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams):
        self.predict_node_info[p_node.client_ip_port]["mem_len"] += len(prompt)
        # self.predict_node_info[p_node.client_ip_port]["batch_size"] += 1
        self.predict_node_info[d_node.client_ip_port]["mem_len"] += sampling_params.max_new_tokens
        # self.predict_node_info[d_node.client_ip_port]["batch_size"] += 1

    def get_predict_node_infos(self):
        return {k: {
            "mem_len": v.get("mem_len", float("inf")),
            # "batch_size": v.get("batch_size", float("inf")),
        } for k, v in self.predict_node_info.items()}

    def get_predict_node_info(self, client_ip_port: str):
        return self.predict_node_info.get(client_ip_port, None)
