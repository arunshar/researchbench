# Dockerfile for ResearchBench Green Agent
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/agentbeats/ ./

EXPOSE 8000

CMD ["python", "-c", "import uvicorn; from green_agent import create_app; uvicorn.run(create_app(), host='0.0.0.0', port=8000)"]
