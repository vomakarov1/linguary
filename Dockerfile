FROM python:3.15-slim

# Install the uv dependency manager at a fixed version
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

# EPUBCheck runs through Java. Install the JRE and the official archive
# of the checking tool, with checksum verification.
ARG EPUBCHECK_VERSION=5.4.0
ARG EPUBCHECK_SHA256=33350c61038e71dfb3d45a76aed04bf5481e6d5500cb780f6e98db8bbd15a28c
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates default-jre-headless \
    && python -c 'import hashlib, io, sys, urllib.request, zipfile; version, expected = sys.argv[1:]; url = f"https://github.com/w3c/epubcheck/releases/download/v{version}/epubcheck-{version}.zip"; archive = urllib.request.urlopen(url, timeout=120).read(); actual = hashlib.sha256(archive).hexdigest(); assert actual == expected, f"EPUBCheck checksum mismatch: {actual}"; zipfile.ZipFile(io.BytesIO(archive)).extractall("/usr/share/java/epubcheck")' "$EPUBCHECK_VERSION" "$EPUBCHECK_SHA256" \
    && java -jar "/usr/share/java/epubcheck/epubcheck-${EPUBCHECK_VERSION}/epubcheck.jar" --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Create a system group and an unprivileged user to run the service
RUN groupadd -r linguary && useradd -r -g linguary -d /app -s /sbin/nologin linguary

WORKDIR /app

# Copy the dependency manifests
COPY pyproject.toml uv.lock ./

# Install project dependencies without dev-only packages and without keeping the cache
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# Copy the source tree and the alembic migrations
COPY alembic alembic
COPY alembic.ini ./
COPY src src
COPY README.md ./

# Install the application package in editable mode without dev-only packages
RUN uv sync --frozen --no-dev --no-cache

# Default environment settings for running inside the container
ENV PATH="/app/.venv/bin:$PATH"
ENV LINGUARY_SERVER_HOST=0.0.0.0
ENV LINGUARY_SERVER_PORT=8000
ENV LINGUARY_STORAGE_DATA_ROOT=/data

# Create the data directory and grant it to the linguary user
RUN mkdir -p /data && chown -R linguary:linguary /app /data

EXPOSE 8000

VOLUME ["/data"]

USER linguary

CMD ["python", "-m", "linguary.cli", "serve"]
