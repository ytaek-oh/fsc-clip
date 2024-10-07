FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update  && \
     apt-get install -y git wget sudo ninja-build nano zip 

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# install a few dependencies
RUN pip install --user jupyterlab
