# syntax = docker/dockerfile:1.2
ARG UBUNTU_RELEASE=20.04
ARG SOURCE_DIR=/home/app/

FROM ubuntu:$UBUNTU_RELEASE
ARG SOURCE_DIR

ARG BOOST_VERSION=1.72.0
ARG BOOST_VERSION_=1.72.0

ENV SOURCE_DIR $SOURCE_DIR
ENV PATH $PATH:$SOURCE_DIR

ENV BOOST_VERSION=${BOOST_VERSION}
ENV BOOST_VERSION_=${BOOST_VERSION_}
ENV BOOST_ROOT=/usr/include/boost



RUN mkdir -p $SOURCE_DIR
WORKDIR $SOURCE_DIR
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --shell /bin/bash --create-home app \
 # install pkgs
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    # you might need build-essential
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    # other pkgs...
 && rm -rf /var/lib/apt/lists/*
# make some useful symlinks
RUN cd /usr/local/bin \
 && ln -s /usr/bin/python3 python \
 && ln -s /usr/bin/python3-config python-config
COPY --chown=app:app ./requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN apt-get -qq update && apt-get install -qy g++ gcc git wget make
RUN cd /usr/local/bin && wget https://sourceforge.net/projects/boost/files/boost/1.72.0/boost_1_72_0.tar.gz \
  && tar xfz boost_1_72_0.tar.gz \
  && rm boost_1_72_0.tar.gz \
  && cd boost_1_72_0 \
  && ./bootstrap.sh --prefix=/usr/local --with-libraries=program_options \
  && ./b2 install


COPY --chown=app:app ./*.sh ./
COPY --chown=app:app ./src/ ./src/
USER app
CMD ["/bin/bash"]
