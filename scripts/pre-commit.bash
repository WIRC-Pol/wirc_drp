#!/usr/bin/env python

import sys

print("Running pre-commit hook")

try:
    import wirc_drp
except Exception as e:
    print("Pre-commit test falied with error {}".format(e))
    sys.exit(-1)

print("Test passed")
sys.exit(0)
