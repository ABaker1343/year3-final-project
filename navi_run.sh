#!/bin/sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 python bnn.py --precision full --no-half --medvram --skip-torch-cuda-test
