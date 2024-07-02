#!/bin/sh

echo "Building container..."

docker build --build-arg UID=$(id -u) -t llm_server_llamadoc .

if [ $? -ne 0 ]; then
        echo "Failed to build container."
        exit 1
fi

echo "Container built."
