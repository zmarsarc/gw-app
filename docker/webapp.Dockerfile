FROM python:3.11-alpine

RUN mkdir -p /app
WORKDIR /app

COPY webapp/ .
COPY gw ./gw

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN pip install -r gw/requirements.txt --no-deps --no-cache-dir
RUN pip install -r requirements.txt --no-deps --no-cache-dir

CMD [ "uvicorn", "--host", "0.0.0.0", "--port", "80", "app.app:app" ]