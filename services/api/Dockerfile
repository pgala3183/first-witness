FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# torch + torchvision first, from TORCH_INDEX. Default PyPI wheels carry CUDA
# (GPU VM). CPU-only VM: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://pypi.org/simple
COPY requirements.txt ./
RUN grep -E '^(torch|torchvision)==' requirements.txt \
        | xargs pip install --no-cache-dir --index-url "$TORCH_INDEX" \
    && pip install --no-cache-dir -r requirements.txt \
    # ultralytics pulls the GUI opencv-python; both own cv2, keep only headless.
    && pip uninstall -y opencv-python \
    && grep -E '^opencv-python-headless==' requirements.txt \
        | xargs pip install --no-cache-dir --no-deps --force-reinstall

COPY contracts ./contracts
COPY services ./services
COPY data ./data
COPY bench ./bench
COPY web ./web
ENV PYTHONPATH=/app
EXPOSE 8080
CMD ["python", "-m", "services.api", "--host", "0.0.0.0", "--port", "8080"]
