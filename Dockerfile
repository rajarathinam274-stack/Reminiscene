FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV REMINISCENCE_DATA_DIR=/data/reminiscence
ENV REMINISCENCE_DB_PATH=/data/reminiscence/reminiscence.db
ENV REMINISCENCE_MODEL_DIR=/data/reminiscence/models
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt README.md LICENSE* ./
COPY reminiscence ./reminiscence
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 reminiscence && mkdir -p /data/reminiscence && chown -R reminiscence:reminiscence /app /data/reminiscence
USER reminiscence
ENTRYPOINT ["reminiscence"]
CMD ["status"]
