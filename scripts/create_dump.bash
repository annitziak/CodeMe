#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: Missing arguments"
    echo "Usage: $0 <output_path> <database_name>"
    exit 1
fi

OUTPUT_PATH="$1"
DB_NAME="$2"

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
fi

if [ ! -w "$OUTPUT_PATH" ]; then
    echo "Error: Directory $OUTPUT_PATH is not writable"
    exit 1
fi

sed "s#output_path#$OUTPUT_PATH#g" create_dump.sql > create_dump_with_path.sql

psql -d "${DB_NAME}" -f create_dump_with_path.sql

rm create_dump_with_path.sql
