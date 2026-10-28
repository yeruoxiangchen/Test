FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 KB_DATA_DIR=/var/lib/knowledge-desk TMPDIR=/var/lib/knowledge-desk/tmp
ENV TORCH_HOME=/var/lib/knowledge-desk/cache/torch HF_HOME=/var/lib/knowledge-desk/cache/huggingface HF_HUB_CACHE=/var/lib/knowledge-desk/cache/huggingface/hub TORCH_EXTENSIONS_DIR=/var/lib/knowledge-desk/cache/torch_extensions MPLCONFIGDIR=/var/lib/knowledge-desk/cache/matplotlib XDG_CACHE_HOME=/var/lib/knowledge-desk/cache
WORKDIR /app
COPY pyproject.toml ./
COPY kb ./kb
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home app && mkdir -p /var/lib/knowledge-desk/tmp && chown -R app:app /var/lib/knowledge-desk
USER app
EXPOSE 8768
CMD ["python", "-m", "uvicorn", "kb.app:app", "--host", "0.0.0.0", "--port", "8768", "--workers", "1"]
