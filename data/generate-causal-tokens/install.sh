pip install -U pip setuptools wheel
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.51.1 pillow accelerate einops safetensors
pip install addict matplotlib
hf download deepseek-ai/DeepSeek-OCR-2 --local-dir ./DeepSeek-OCR-2-weights
