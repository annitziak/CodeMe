#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Error: Missing arguments"
    echo "Usage: $0 <output_path> <database_name> <number_of_posts>"
    exit 1
fi

OUTPUT_PATH="$1"
DB_NAME="$2"
NUMBER_OF_POSTS="$3"

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
fi

if [ ! -w "$OUTPUT_PATH" ]; then
    echo "Error: Directory $OUTPUT_PATH is not writable"
    exit 1
fi

sed "s#output_path#$OUTPUT_PATH#g" create_dump.sql > temp1.sql
sed "s#db_name#$DB_NAME#g" temp1.sql > temp2.sql
sed "s#number_of_posts#$NUMBER_OF_POSTS#g" temp2.sql > create_dump_with_path.sql

rm temp1.sql
rm temp2.sql

psql -d "${DB_NAME}" -f create_dump_with_path.sql

rm create_dump_with_path.sql
