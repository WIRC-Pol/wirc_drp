#!/usr/bin/env bash

GIT_DIR=$(git rev-parse --git-dir)

echo "Running pre-commit hook\n"

echo "Starting Test 1"
python tests/test1.py

if [ $? -ne 0 ]; then
 echo "Pre-commit tests failed on test 1"
 exit 1
fi

echo "Test 1 passed"
echo ""

echo "Starting Test 2"
python tests/test2.py

if [ $? -ne 0 ]; then
 echo "Pre-commit tests failed on test 2"
 exit 1
fi

echo "Test 2 passed"
echo ""

echo "Starting Test 3"
python tests/test3.py

if [ $? -ne 0 ]; then
 echo "Pre-commit tests failed on test 3"
 exit 1
fi

echo "Test 3 passed"
echo ""

