FROM python:3.9.15-slim-buster

WORKDIR /app

COPY requirements.txt . 
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000

RUN ["uvicorn", "main:app"]


