# CPU controller for API or existing self-hosted endpoints; sandbox jobs use the host daemon.
FROM docker:27-cli AS docker-cli
FROM python:3.12-slim-bookworm
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/si2ca
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE ./
COPY si2ca ./si2ca
COPY configs ./configs
COPY skills ./skills
COPY data/*.jsonl data/*.csv data/assets.tar.gz ./data/
ARG SI2CA_EXTRAS=search
RUN pip install --no-cache-dir ".[${SI2CA_EXTRAS}]"
ENV SI2CA_CACHE_DIR=/cache/si2ca PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["si2ca"]
CMD ["--help"]
