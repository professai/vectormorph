FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir vectormorph

EXPOSE 4440

CMD ["vectormorph"]
