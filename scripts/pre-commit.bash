#!/usr/bin/env bash

GIT_DIR=$(git rev-parse --git-dir)

echo "Running pre-commit hook\n"

echo "Starting Test 1"
python tests/test1.py

if [ $? -ne 0 ]; then
 echo "Pre-commit tests failed on test 1"
 exit 1
fi

