FROM python:3.11-alpine

RUN mkdir -p /app
WORKDIR /app

COPY postprocess/  .
COPY gw ./gw

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install -r gw/requirements.txt --no-deps --no-cache-dir

CMD [ "python", "app.py" ]