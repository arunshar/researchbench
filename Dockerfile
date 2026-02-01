FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY README.md .

ENV PYTHONUNBUFFERED=1

# Green agent on port 8000
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.green_agent:create_app", "--host", "0.0.0.0", "--port", "8000"]
