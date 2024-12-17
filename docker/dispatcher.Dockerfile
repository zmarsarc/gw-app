FROM gw/ascendbase:latest

USER 0
RUN mkdir -p /app
WORKDIR /app

COPY dispatcher/  .
COPY gw ./gw
COPY gwmodel ./gwmodel

RUN chown -R HwHiAiUser:HwHiAiUser /app

USER 1000
ENV PATH=/home/AscendWork/miniconda/bin:${PATH}
RUN conda create --name gw -y python=3.11

ENV CONDA_DEFAULT_ENV=gw
RUN echo "conda activate gw" >> ~/.bashrc

# If build too slow in mainland, use aliyun mirros when need.
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install dependences.
RUN pip install -r gw/requirements.txt --no-deps --no-cache-dir
RUN pip install -r gwmodel/requirements.txt --no-deps --no-cache-dir

CMD ["python", "task_dispatcher.py"]
