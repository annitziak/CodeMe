#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: Missing arguments"
    echo "Usage: $0 <input_path> <database_name>"
    exit 1
fi

INPUT_PATH="$1"
DB_NAME="$2"

if [ ! -d "$INPUT_PATH" ]; then
    mkdir -p "$OU"
fi

if [ ! -w "$INPUT_PATH" ]; then
    echo "Error: Directory $INPUT_PATH is not writable"
    exit 1
fi

sed "s#input_path#$INPUT_PATH#g" read_dump.sql > create_dump_with_path.sql

echo "Running the following command:"
echo "psql -d ${DB_NAME} -f create_dump_with_path.sql"

psql -d "${DB_NAME}" -f create_dump_with_path.sql

rm create_dump_with_path.sql
