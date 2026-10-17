FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    LOGFIRE_IGNORE_NO_CONFIG=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Bake both models into the image so cold starts don't download them
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/tmp/flashrank')"
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY src ./src
COPY main.py .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
