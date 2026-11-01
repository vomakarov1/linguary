FROM python:3.13-slim

# Установка инструмента управления зависимостями uv с фиксированной версией
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

# EPUBCheck запускается через Java. Устанавливаем JRE и официальный архив
# проверяющего инструмента с проверкой контрольной суммы.
ARG EPUBCHECK_VERSION=5.4.0
ARG EPUBCHECK_SHA256=33350c61038e71dfb3d45a76aed04bf5481e6d5500cb780f6e98db8bbd15a28c
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates default-jre-headless \
    && python -c 'import hashlib, io, sys, urllib.request, zipfile; version, expected = sys.argv[1:]; url = f"https://github.com/w3c/epubcheck/releases/download/v{version}/epubcheck-{version}.zip"; archive = urllib.request.urlopen(url, timeout=120).read(); actual = hashlib.sha256(archive).hexdigest(); assert actual == expected, f"EPUBCheck checksum mismatch: {actual}"; zipfile.ZipFile(io.BytesIO(archive)).extractall("/usr/share/java/epubcheck")' "$EPUBCHECK_VERSION" "$EPUBCHECK_SHA256" \
    && java -jar "/usr/share/java/epubcheck/epubcheck-${EPUBCHECK_VERSION}/epubcheck.jar" --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Создание системной группы и непривилегированного пользователя для запуска сервиса
RUN groupadd -r linguary && useradd -r -g linguary -d /app -s /sbin/nologin linguary

WORKDIR /app

# Копирование манифестов зависимостей
COPY pyproject.toml uv.lock ./

# Установка зависимостей проекта без dev-пакетов и без сохранения кэша
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# Копирование дерева исходного кода и миграций alembic
COPY alembic alembic
COPY alembic.ini ./
COPY src src
COPY README.md ./

# Установка пакета приложения в редактируемом режиме без dev-пакетов
RUN uv sync --frozen --no-dev --no-cache

# Настройки окружения по умолчанию для работы внутри контейнера
ENV PATH="/app/.venv/bin:$PATH"
ENV LINGUARY_SERVER_HOST=0.0.0.0
ENV LINGUARY_SERVER_PORT=8000
ENV LINGUARY_STORAGE_DATA_ROOT=/data

# Создание каталога данных и назначение прав пользователю linguary
RUN mkdir -p /data && chown -R linguary:linguary /app /data

EXPOSE 8000

VOLUME ["/data"]

USER linguary

CMD ["python", "-m", "linguary.cli", "serve"]
