#!/bin/bash
pip install --upgrade pip wheel setuptools
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
