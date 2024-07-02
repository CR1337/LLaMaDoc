#!/bin/sh

export GROUP=2
export GPU=$((AI_GROUP % 4))
export CACHE_DIR=/var/opt/huggingface
export PORT=$((9000 + GROUP))

echo "Creating container..."

docker run --rm --init --name ai4se-2-llm-server --gpus device=$GPU \
        -p 0.0.0.0:$PORT:8000 -v $CACHE_DIR:/root/.cache/huggingface llm_server_llamadoc

if [ $? -ne 0 ]; then
        echo "Failed to create container."
        exit 1
fi

echo "Container created."
