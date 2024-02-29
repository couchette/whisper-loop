FROM ubuntu:22.04

WORKDIR /app

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean && apt-get update && apt-get install -y python3 python3-pip ffmpeg patchelf portaudio19-dev
COPY ./requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
