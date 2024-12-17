FROM ubuntu:22.04

ARG ASCEND_BASE=/usr/local/Ascend
WORKDIR /home/AscendWork

# 推理程序需要使用到底层驱动，底层驱动的运行依赖HwHiAiUser，HwBaseUser，HwDmUser三个用户
# 创建运行推理应用的用户及组，HwHiAiUse，HwDmUser，HwBaseUser的UID与GID分别为1000，1101，1102为例
RUN groupadd  HwHiAiUser -g 1000 && \
    useradd -d /home/HwHiAiUser -u 1000 -g 1000 -m -s /bin/bash HwHiAiUser

RUN sed -i "s@http://.*ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list && \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends git python3 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN --mount=type=cache,target=/tmp,from=gw/ascenddeps,source=/ascend \
    bash /tmp/Ascend-cann-nnrt_7.0.1_linux-aarch64.run --quiet --install --install-path=$ASCEND_BASE --install-for-all --force && \
    bash /tmp/Ascend-cann-toolkit_7.0.1_linux-aarch64.run --quiet --install --install-path=$ASCEND_BASE --install-for-all --force
    
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

# 安装conda，使用conda管理python版本
RUN --mount=type=cache,target=/tmp,from=gw/ascenddeps,source=/ascend \
    bash /tmp/Anaconda3-2024.10-1-Linux-aarch64.sh -b -p ./miniconda && \
    ./miniconda/bin/conda init

# 设置环境
ENV DDK_PATH=${ASCEND_BASE}/ascend-toolkit/latest
ENV TOOLCHAIN_HOME=${DDK_PATH}/toolkit
ENV NPU_HOST_LIB=${DDK_PATH}/runtime/lib64/stub
ENV THIRDPART_PATH=${DDK_PATH}/thirdpart
ENV PYTHONPATH=${THIRDPART_PATH}/python:${DDK_PATH}/python/site-packages:${DDK_PATH}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV LD_LIBRARY_PATH=${ASCEND_BASE}driver/lib64/common:${ASCEND_BASE}/driver/lib64/driver:${LD_LIBRARY_PATH}

RUN mkdir -p ${THIRDPART_PATH}
RUN --mount=type=cache,target=/tmp,from=gw/ascenddeps,source=/ascend \
    cp -r /tmp/ACLLite/python ${THIRDPART_PATH}

USER 1000
RUN ./miniconda/bin/conda init

CMD [ "/bin/bash" ]
