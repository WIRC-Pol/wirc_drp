#!/usr/bin/env bash

echo "Running pre-commit hook\n"

echo "Starting Test 1"
python ./scripts/test1.py

if [ $? -ne 0 ]; then
 echo "Pre-commit tests failed on test 1"
 exit 1
fi

