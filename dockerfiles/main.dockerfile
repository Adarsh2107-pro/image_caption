# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Ignore data to keep the image reasonably small
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY image_caption/ image_caption/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Run as module to avoid import errors
ENTRYPOINT ["python", "-m", "image_caption.main"]
