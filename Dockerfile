FROM python:3.13-slim

# Установка инструмента управления зависимостями uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копирование манифестов зависимостей
COPY pyproject.toml uv.lock ./

# Установка зависимостей проекта без установки самого пакета для кэширования слоёв
RUN uv sync --frozen --no-install-project

# Копирование дерева исходного кода и миграций alembic
COPY alembic alembic
COPY alembic.ini ./
COPY src src
COPY README.md ./

# Установка пакета приложения в редактируемом режиме
RUN uv sync --frozen

# Настройки окружения по умолчанию для работы внутри контейнера
ENV PATH="/app/.venv/bin:$PATH"
ENV LINGUARY_SERVER_HOST=0.0.0.0
ENV LINGUARY_SERVER_PORT=8000
ENV LINGUARY_STORAGE_DATA_ROOT=/data

# Создание каталога данных
RUN mkdir -p /data

EXPOSE 8000

VOLUME ["/data"]

CMD ["python", "-m", "linguary.cli", "serve"]
