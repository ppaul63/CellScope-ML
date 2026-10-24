FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ARTIFACTS_DIR=/srv/artifacts
COPY backend/requirements.txt backend/requirements.txt
# CPU-only torch keeps the image small; it is only used for Grad-CAM heatmaps
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements.txt
COPY ml/ ml/
COPY backend/ backend/
COPY models.yaml models.yaml
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
