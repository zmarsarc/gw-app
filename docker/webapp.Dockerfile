FROM python:3.11-alpine

RUN mkdir -p /app
WORKDIR /app

COPY webapp/ .
COPY gw ./gw

RUN pip install -r requirements.txt

CMD [ "uvicorn", "--host", "0.0.0.0", "--port", "80", "app.app:app" ]