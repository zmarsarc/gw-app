FROM ubuntu:22.04

RUN mkdir -p /ascend
WORKDIR /ascend

COPY . ./
RUN sed -i "s@http://.*ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN apt update && apt install -y git
RUN git clone https://gitee.com/ascend/ACLLite.git

CMD [ "/bin/bash" ]