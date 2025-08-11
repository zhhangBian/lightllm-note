import random
import pytest
import torch

from lightllm.common.kv_trans_kernel.kv_trans_v2 import (
    kv_trans_v2_for_p_node,
    kv_trans_v2_for_d_node,
)


@pytest.mark.parametrize("token_num", [t for t in range(5, 10)])
def test_kv_trans_v2_for_p_node_double_heads(token_num: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    # 模拟 MemoryManager 使用的形状: (token_num, 2 * head_num, head_dim)
    dp_size_in_node = 3
    head_num, head_dim, kv_buffer_token_num = 2, 64, 256
    double_head_num = 2 * head_num

    # 为每个 dp 副本构造一段 kv 缓存 (kv_buffer_token_num, 2*head_num, head_dim)
    mems = [
        torch.randn((kv_buffer_token_num, double_head_num, head_dim), dtype=torch.float16, device="cuda")
        for _ in range(dp_size_in_node)
    ]
    input_mems = torch.tensor([m.data_ptr() for m in mems], dtype=torch.uint64, device="cuda")

    # 随机采样 token 索引与其所属 dp 索引
    input_idx = torch.tensor(
        [random.randint(0, kv_buffer_token_num - 1) for _ in range(token_num)], dtype=torch.int32, device="cuda"
    )
    input_dp_idx = torch.tensor(
        [random.randint(0, dp_size_in_node - 1) for _ in range(token_num)], dtype=torch.int32, device="cuda"
    )

    # 输出缓冲区: (token_num, 2*head_num, head_dim)
    output = torch.zeros((token_num, double_head_num, head_dim), dtype=torch.float16, device="cuda")
    output_idx = torch.arange(0, token_num, 1, dtype=torch.int32, device="cuda")

    kv_trans_v2_for_p_node(input_mems, input_idx, input_dp_idx, output, output_idx, dp_size_in_node)

    # 校验抽取出的 KV 数据与期望一致
    expected = torch.zeros_like(output)
    for dest_i, (tok_i, dp_i) in enumerate(zip(input_idx.cpu().tolist(), input_dp_idx.cpu().tolist())):
        expected[dest_i] = mems[dp_i][tok_i]

    assert torch.equal(output, expected)


@pytest.mark.parametrize("token_num", [t for t in range(5, 10)])
def test_kv_trans_v2_for_d_node_replication(token_num: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    # 在 decode 端，内核会把同一 token 的数据复制到该 dp 组内的所有卡上
    dp_size_in_node = 2
    cards_per_dp = 3  # 每个 dp 组包含的卡数
    total_cards = dp_size_in_node * cards_per_dp

    head_num, head_dim, kv_buffer_token_num = 2, 64, 256
    double_head_num = 2 * head_num

    # 输出 mems: 每张卡一块 KV 缓存 (kv_buffer_token_num, 2*head_num, head_dim)
    mems = [
        torch.zeros((kv_buffer_token_num, double_head_num, head_dim), dtype=torch.float16, device="cuda")
        for _ in range(total_cards)
    ]
    output_mems = torch.tensor([m.data_ptr() for m in mems], dtype=torch.uint64, device="cuda")

    # 输入: 要被写入的 token 数据 (token_num, 2*head_num, head_dim)
    input_tensor = torch.randn((token_num, double_head_num, head_dim), dtype=torch.float16, device="cuda")
    input_idx = torch.arange(0, token_num, 1, dtype=torch.int32, device="cuda")

    # 目标 token 在各卡 KV 中的位置, 以及所属 dp 组索引
    output_idx = torch.tensor(
        [random.randint(0, kv_buffer_token_num - 1) for _ in range(token_num)], dtype=torch.int32, device="cuda"
    )
    output_dp_idx = torch.tensor(
        [random.randint(0, dp_size_in_node - 1) for _ in range(token_num)], dtype=torch.int32, device="cuda"
    )

    kv_trans_v2_for_d_node(output_mems, output_idx, output_dp_idx, input_tensor, input_idx, dp_size_in_node)

    # 校验: 每个 token 应复制到其 dp 组内所有卡的指定行
    for t_i, (dst_row, dp_i) in enumerate(zip(output_idx.cpu().tolist(), output_dp_idx.cpu().tolist())):
        start = dp_i * cards_per_dp
        end = (dp_i + 1) * cards_per_dp
        for mem_i in range(start, end):
            assert torch.equal(mems[mem_i][dst_row], input_tensor[t_i])


if __name__ == "__main__":
    pytest.main([__file__]) 