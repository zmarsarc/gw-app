FROM python:3.11-slim

RUN mkdir -p /app
WORKDIR /app

COPY dispatcher/  .
COPY gw ./gw
COPY gwmodel ./gwmodel

# If build too slow in mainland, use aliyun mirros when need.
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install dependences.
RUN pip install -r gw/requirements.txt --no-deps --no-cache-dir
RUN pip install -r gwmodel/requirements.txt --no-deps --no-cache-dir

CMD [ "python", "task_dispatcher.py" ]