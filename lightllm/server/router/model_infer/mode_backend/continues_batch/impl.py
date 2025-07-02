import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample

logger = init_logger(__name__)


# 连续批处理实现
class ContinuesBatchBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        # 将当前获取的运行请求进行分类，获取五类请求
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        # 将中止请求进行释放
        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        # 对需要prefill的请求进行处理
        if prefill_reqs:
            # 对prefill请求进行处理，准备输入的数据
            model_input, run_reqs = prepare_prefill_inputs(
                prefill_reqs, is_chuncked_mode=False, is_multimodal=self.is_multimodal
            )
            # 对prefill请求进行处理，准备输入的数据
            model_output = self.model.forward(model_input)
            # 对prefill请求进行处理，准备输入的数据
            logits = model_output.logits

            # 将未初始化请求和已完成请求进行合并，并释放未初始化请求
            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            # 对decode请求进行后处理
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

        # 对需要decode的请求进行处理
        if decode_reqs:
            # 对decode请求进行处理，准备输入的数据
            model_input, run_reqs = prepare_decode_inputs(decode_reqs)
            model_output = self.model.forward(model_input)
            logits = model_output.logits

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

        # 对未初始化请求和已完成请求进行合并，并释放未初始化请求  
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
