FROM ubuntu:18.04
RUN apt-get update \
 && apt-get install --no-install-recommends -y gcc
RUN mkdir -p /opt/darknet
COPY . /opt/darknet
WORKDIR /opt/darknet
RUN make -j8
