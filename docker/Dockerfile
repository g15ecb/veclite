FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang libclang-rt-18-dev python3 python3-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN pip install --no-cache-dir mojo==1.2.0 max==26.7.0 numpy pytest
WORKDIR /src
COPY . .
CMD ["scripts/check.sh"]
