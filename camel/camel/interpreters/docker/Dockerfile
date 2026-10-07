# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    wget \
    git \
    git-lfs \
    netcat-openbsd \
    sudo \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install uv

RUN ln -sf /opt/venv/bin/python /usr/bin/python

RUN apt-get update && \
    apt-get install -y r-base && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get autoremove -y

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get autoremove -y

RUN curl -fsSL https://install.python-poetry.org | python - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

# Pre-install common packages for GAIA benchmark
RUN pip install --no-cache-dir \
    markitdown \
    PyPDF2 \
    openpyxl \
    pandas \
    requests \
    beautifulsoup4 \
    lxml \
    python-docx \
    Pillow \
    numpy \
    pytz

RUN groupadd -r devuser && useradd -r -m -g devuser devuser
WORKDIR /workspace
RUN chown -R devuser:devuser /workspace /opt/venv
USER devuser

CMD ["/bin/bash"]
