FROM python:3.10-buster

COPY ./ ./app/
WORKDIR /app

WORKDIR /app

RUN apt-get update -y
RUN pip install --upgrade pip

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80