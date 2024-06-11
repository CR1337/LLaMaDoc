#!/bin/sh

uvicorn server:server --host "0.0.0.0" --port 8000 --reload
