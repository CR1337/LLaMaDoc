#!/bin/sh

export GROUP=2
export GPU=$((AI_GROUP % 4))
export CACHE_DIR=/var/opt/huggingface
export PORT=$((9000 + GROUP))

echo "Creating container..."

docker run -d --name ai4se-2-llm-server --gpus "device=$GPU" \
        -p 0.0.0.0:$PORT:8000 -v $CACHE_DIR:/root/.cache/huggingface llm_server_llamadoc

if [ $? -ne 0 ]; then
        echo "Failed to create container."
        exit 1
fi

echo "Container created."


#------------------------------------------------------------------------

# export AI_GROUP=2
# export AI_LOGIN=

# if [[ $# -eq 1 ]] ; then
#         echo "Using group $1 from command line"
#         export AI_GROUP=$1;
# fi

# if [[ $# -eq 2 ]] ; then
#         echo "Using group $1 and login token from command line"
#         export AI_GROUP=$1
#         export AI_LOGIN=$2;
# fi

# # This is automatically computed
# export AI_GPU=$((AI_GROUP % 4))
# export AI_PORT=$((8000 + AI_GROUP))
# export AI_CONTAINER_NAME=ai4se-$AI_GROUP

# export AI_CACHE_DIR=/var/opt/huggingface

# echo "Creating container for group $AI_GROUP using GPU $AI_GPU at port $AI_PORT"

# docker run -d --name ai4se-$AI_GROUP --gpus "device=$AI_GPU" -p 0.0.0.0:$AI_PORT:8888 \
#         -v ~/workspace:/workspace \
#         -v $AI_CACHE_DIR:/root/.cache/huggingface \
#         -e LOGIN=$AI_LOGIN \
#         ai4se

# echo "URL: http://delos.eaalab.hpi.uni-potsdam.de:$AI_PORT/lab"
# echo "This container is named $AI_CONTAINER_NAME"
# echo "Stop: docker stop $AI_CONTAINER_NAME"
# echo "Start: docker start $AI_CONTAINER_NAME"
# echo "Remove: docker rm $AI_CONTAINER_NAME"
# echo "Rebuild: (run this script)"
