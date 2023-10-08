FROM python:3.10-buster

ENV BEARER_TOKEN=1234
COPY ./ ./app/
WORKDIR /app
RUN pip install -U vectormorph

EXPOSE 4440