FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# The model is not baked in: artifacts/ is mounted at runtime (see docker-compose.yml).
RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "credit.api:app", "--host", "0.0.0.0", "--port", "8000"]
