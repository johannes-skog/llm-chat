FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update 

RUN apt-get -y install \
    curl \
    wget \
    build-essential \
    screen \
    git \ 
    htop

# install az-cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY entrypoint.sh /tmp
ENTRYPOINT ["bash", "/tmp/entrypoint.sh"]

ENV PATH="${PATH}:/opt/conda/bin/"

