CUDA_VISIBLE_DEVICES=1 python \
    -m lightllm.server.api_server \
    --model_dir /home/youwei/bzh/model/opendatalab/MinerU2.5-2509-1.2B \
    --host 0.0.0.0 \
    --port 8087 \
    --enable_multimodal \
    --max_req_total_len 18432 \
    --enable_fa3

curl http://127.0.0.1:8081/generate \
    -H "Content-Type: application/json" \
    -d '{
          "inputs": "What is AI?",
          "parameters":{
            "max_new_tokens":128,
            "frequency_penalty":1
          }
        }'

curl http://127.0.0.1:8081/health

