#!/usr/bin/env bash
pip install --upgrade pip wheel setuptools
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
pip install -r requirements.txt
