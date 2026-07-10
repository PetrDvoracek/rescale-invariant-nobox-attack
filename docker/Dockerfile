# Named CUDA/torch base — matches .devcontainer/Dockerfile (not `latest`).
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

# The NGC base ships a custom-compiled 'opencv' that breaks albumentations'
# import; fix it before installing the curated pins (mirrors .devcontainer).
RUN pip install --upgrade "pip<25" \
    && pip install opencv-fixer==0.2.5 \
    && python -c "from opencv_fixer import AutoFix; AutoFix()"

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
