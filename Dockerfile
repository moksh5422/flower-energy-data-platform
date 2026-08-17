FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY data ./data
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["sh","-c","python -m flower_pipeline.generate_data && python -m flower_pipeline.pipeline && python -m flower_pipeline.train && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
