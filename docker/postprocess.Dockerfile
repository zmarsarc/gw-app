FROM python:3.11-alpine

RUN mkdir -p /app
WORKDIR /app

COPY postprocess/  .
COPY gw ./gw

RUN pip install -r requirements.txt

CMD [ "python", "app.py" ]