FROM python:3.12-slim

# Опціонально — щоб уникнути питань з TLS і часовими зонами
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Встановлюємо системні залежності (можна розширити при потребі)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо Python-залежності
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    torch==2.3.0 \
    cloudpickle==3.0.0 \
    requests==2.32.3 \
    numpy \ 
    tqdm

# Додаємо код у контейнер
WORKDIR /app
COPY . .

# Запускаємо сервер
CMD ["python", "coordinator.py"]  