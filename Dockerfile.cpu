FROM ubuntu:18.04
RUN apt-get update \
 && apt-get install --no-install-recommends -y gcc build-essential
RUN mkdir -p /opt/darknet
RUN mkdir -p /opt/bin
COPY . /opt/darknet
WORKDIR /opt/darknet
RUN make -j8
RUN cp darknet /opt/bin
