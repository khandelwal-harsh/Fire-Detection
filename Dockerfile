from ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
RUN apt update 
RUN apt-get install wget -y
RUN  wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1_79BO0PhIW_6-RX41nOzxR0B8jTKoGEk" -O firenet_v2.hdf5 -r -A 'uc*' -e robots=off -nd
RUN ls -lh
RUN ls /src -lh
COPY . /src
RUN ls /src -lh
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get install python3-h5py
RUN apt update
RUN apt-get install -y python3-opencv

RUN pip3 install flask opencv-python
RUN pip3 install --upgrade tensorflow
RUN pip3 install keras
RUN pip3 install pillow
RUN pip3 install tqdm






