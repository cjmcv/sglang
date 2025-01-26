#!/usr/bin/env bash

# sh run_test.sh sglang 0

MODEL_PATH="/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct"

DATASET_PATH=/home/cjmcv/project/llm_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
DATASET="sharegpt"

NSYS_PROFILER=
# NSYS_PROFILER="nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70"
NUM_PROMPTS=10
REQUEST_RATE=4

if [ "$2" = "0" ]; then
    if [ "$1" = "sglang" ]; then
        # --grammar-backend xgrammar --disable-overlap-schedule --disable-radix-cache
        $NSYS_PROFILER python3 -m sglang.launch_server --model-path $MODEL_PATH --enable-torch-compile --enable-mixed-chunk 
    elif [ "$1" = "vllm" ]; then
        $NSYS_PROFILER python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --disable-log-requests --num-scheduler-steps 10 --max_model_len 4096
    elif [ "$1" = "lmdeploy" ]; then
        $NSYS_PROFILER lmdeploy serve api_server $MODEL_PATH  --server-port 23333 --model-name test_model
    else
        echo "abc"
    fi
elif [ "$2" = "1" ]; then
    python3 -m benchmark --backend $1 --dataset-name $DATASET --dataset-path $DATASET_PATH --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE
elif [ "$2" = "2" ]; then
    nsys-ui profile sglang.out.nsys-rep
fi

## Setup command.
## sgLang
# pip install --upgrade pip
# pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
#
# cd sglang
# pip install --upgrade pip
# pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# curl http://localhost:30000/generate \
#   -H "Content-Type: application/json" \
#   -d '{
#     "text": "Once upon a time,",
#     "sampling_params": {
#       "max_new_tokens": 16,
#       "temperature": 0
#     }
#   }'

# curl http://localhost:30000/start_profile

## vllm
# pip uninstall vllm
# pip install vllm=0.6.6

## lmdeploy
# conda create -n lmdeploy python=3.8 -y
# conda activate lmdeploy
# pip install lmdeploy

# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct-AWQ', cache_dir='/home/cjmcv/project/llm_models/')
# conda create -n eval-venv python=3.10 -y
# conda activate eval-venv
# pip install -e .
# lighteval vllm     "pretrained=/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct-AWQ,dtype=float16"     "leaderboard|truthfulqa:mc|0|0"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True