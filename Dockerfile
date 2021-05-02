from ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src

COPY . /src

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*



RUN apt update
RUN apt-get install -y python3-opencv
RUN pip3 install flask opencv-python
RUN pip3 install --upgrade tensorflow




