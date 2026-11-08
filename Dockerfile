
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies from the lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Copy training and inference scripts
COPY train.py predict.py ./
COPY data/biodeg.csv data/biodeg.csv

# Train and save model.bin inside the image
RUN python train.py

EXPOSE 9696

# Start the Flask API through Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "1", "predict:app"]
