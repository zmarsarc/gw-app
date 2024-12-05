FROM docker.unsee.tech/ubuntu:22.04 as buildtemp
WORKDIR /tmp
COPY . ./
RUN sed -i "s@http://.*ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN apt update && apt install -y git
RUN git clone https://gitee.com/ascend/ACLLite.git

FROM docker.unsee.tech/ubuntu:22.04

ARG ASCEND_BASE=/usr/local/Ascend
WORKDIR /home/AscendWork

# 推理程序需要使用到底层驱动，底层驱动的运行依赖HwHiAiUser，HwBaseUser，HwDmUser三个用户
# 创建运行推理应用的用户及组，HwHiAiUse，HwDmUser，HwBaseUser的UID与GID分别为1000，1101，1102为例
RUN sed -i "s@http://.*ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN apt update && \
    apt upgrade -y && \
    apt install -y git python3 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    umask 0022 && \
    groupadd  HwHiAiUser -g 1000 && \
    useradd -d /home/HwHiAiUser -u 1000 -g 1000 -m -s /bin/bash HwHiAiUser

RUN --mount=type=cache,target=/tmp,from=buildtemp,source=/tmp \
    ls -al /tmp && \
    bash /tmp/Ascend-cann-nnrt*.run --quiet --install --install-path=$ASCEND_BASE --install-for-all --force
    
RUN --mount=type=cache,target=/tmp,from=buildtemp,source=/tmp \
    mkdir -p ~/.pip  && \
    echo '[global] \n\
    index-url=http://mirrors.aliyun.com/pypi/simple\n\
    trusted-host=mirrors.aliyun.com' >> ~/.pip/pip.conf && \
    pip3 install pip -U && \
    pip3 install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip && \
    bash /tmp/Ascend-cann-toolkit*.run --quiet --install --install-path=$ASCEND_BASE --install-for-all --force

RUN chown -R HwHiAiUser:HwHiAiUser /home/AscendWork/ && \
    ln -sf /lib /lib64

ENV LD_LIBRARY_PATH=/usr/local/Ascend/nnrt/latest/lib64:/usr/local/Ascend/driver/lib64:/usr/lib64
ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libc.so.6
	
# 可使用AICPU算子
RUN cp /usr/local/Ascend/nnrt/latest/opp/Ascend/aicpu/Ascend-aicpu_syskernels.tar.gz /home/HwHiAiUser/ && \
    rm -rf /usr/local/Ascend/nnrt/latest/opp/Ascend/aicpu/Ascend-aicpu_syskernels.tar.gz && \
    echo $(wc -c /home/HwHiAiUser/Ascend-aicpu_syskernels.tar.gz|awk ' {print$1} ') > /home/HwHiAiUser/aicpu_package_install.info && \
    tail -c +8449 /home/HwHiAiUser/Ascend-aicpu_syskernels.tar.gz > /home/HwHiAiUser/aicpu.tar.gz && \
    rm -rf /home/HwHiAiUser/Ascend-aicpu_syskernels.tar.gz && \
    chown HwHiAiUser:HwHiAiUser /home/HwHiAiUser/aicpu.tar.gz && \
    mkdir -p /home/HwHiAiUser/aicpu_kernels && \
    tar -xvf /home/HwHiAiUser/aicpu.tar.gz -C /home/HwHiAiUser/ 2>/dev/null || true && \
    rm -rf /home/HwHiAiUser/aicpu.tar.gz && \
    mv /home/HwHiAiUser/aicpu_kernels_device/* /home/HwHiAiUser/aicpu_kernels/ && \
    chown -R HwHiAiUser:HwHiAiUser /home/HwHiAiUser/

RUN pip install torch torchvision numpy opencv-python-headless Pillow Cython av paddlepaddle==2.6.2 onnxruntime shapely pyclipper

ENV DDK_PATH=${ASCEND_BASE}/ascend-toolkit/latest
ENV TOOLCHAIN_HOME=${DDK_PATH}/toolkit
ENV NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
ENV THIRDPART_PATH=${DDK_PATH}/thirdpart
ENV PYTHONPATH=${THIRDPART_PATH}/python:${DDK_PATH}/python/site-packages:${DDK_PATH}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV LD_LIBRARY_PATH=${ASCEND_BASE}driver/lib64/common:${ASCEND_BASE}/driver/lib64/driver:${LD_LIBRARY_PATH}

RUN mkdir -p ${THIRDPART_PATH}
RUN --mount=type=cache,target=/tmp,from=buildtemp,source=/tmp \
    cp -r /tmp/ACLLite/python ${THIRDPART_PATH}

USER 1000
CMD bash /home/AscendWork/run.sh