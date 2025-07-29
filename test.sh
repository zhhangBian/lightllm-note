python3 test/benchmark_client.py \
  --url http://127.0.0.1:60011/generate_stream \
  --tokenizer_path /mtc/bianzhuohang/models/Qwen/Qwen2.5-14B \
  --server_api lightllm \
  --dump_file result.json \
  --seed 42 \
  --input_len 4096 \
  --output_len 1024 \
  --input_num 1000 \
  --dump_file pd_random.json
