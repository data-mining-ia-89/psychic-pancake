FROM python:3.11.9-slim-bookworm

# Reduce vulnerabilities by updating system packages and installing security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY . .

ENV PYTHONPATH=/code

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
