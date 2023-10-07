FROM python:3.10-buster

ENV BEARER_TOKEN=1234
COPY ./ ./app/
WORKDIR /app
RUN pip install vectormorph

EXPOSE 80 4440