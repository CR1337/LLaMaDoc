#!/bin/sh

docker run --rm --name github-scraper-db -p 5432:5432 -v ./db/postgres-data:/var/lib/postgresql/data github-scraper-db
