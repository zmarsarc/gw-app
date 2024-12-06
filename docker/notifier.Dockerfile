FROM python:3.11-alpine

RUN mkdir -p /app
WORKDIR /app

COPY notifier/  .
COPY gw ./gw

# Use aliyun to speed up build process.
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN pip install -r gw/requirements.txt --no-deps --no-cache-dir
RUN pip install -r requirements.txt --no-deps --no-cache-dir

CMD [ "python", "app.py" ]