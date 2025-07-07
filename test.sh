python3 test/benchmark_client.py \
  --url http://127.0.1.1:60011/generate \
  --tokenizer_path /mnt/youwei-data/zhuohang/model/Qwen/Qwen2.5-14B \
  --output_len 16384 \
  --server_api lightllm \
  --dump_file result.json \
  --seed 42