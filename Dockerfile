FROM python:3.13-slim

# Установка инструмента управления зависимостями uv с фиксированной версией
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

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
