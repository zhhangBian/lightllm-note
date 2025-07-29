python3 -m lightllm.server.api_server \
    --model_dir /mtc/bianzhuohang/models/Qwen/Qwen2.5-14B \
    --run_mode "pd_master" \
    --host 127.0.0.1 \
    --port 60011 \
    --select_p_d_node_func random

# curl http://127.0.0.1:60011/generate \
#     -H "Content-Type: application/json" \
#     -d '{
#           "inputs": "What is AI?",
#           "parameters":{
#             "max_new_tokens":128,
#             "frequency_penalty":1
#           }
#         }'