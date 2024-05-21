#!/bin/sh

if [ $# -ne 1 ]; then
    STEP=0
else
    STEP=$1
fi

if [ $STEP -le 0 ]; then
    python3 00_repository_metadata_scraper.py
    if [ $? -ne 0 ]; then exit 1; fi
fi

if [ $STEP -le 1 ]; then
    python3 01_repository_filter.py
    if [ $? -ne 0 ]; then exit 1; fi
fi

if [ $STEP -le 2 ]; then
    python3 02_repository_scraper.py
    if [ $? -ne 0 ]; then exit 1; fi
fi

if [ $STEP -le 3 ]; then
    python3 03_extract_data.py
    if [ $? -ne 0 ]; then exit 1; fi
fi
