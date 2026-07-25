# Production image for the council server.
#
# Deliberately boring: any PaaS that can build a Dockerfile can host this, and
# it is the only route that works when a phone is the only device you have.
FROM python:3.12-slim

# Never write .pyc, never buffer logs (a PaaS reads stdout for its log stream).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Dependencies first so a code change doesn't invalidate the wheel cache.
COPY pyproject.toml README.md ./
COPY orchestra/__init__.py ./orchestra/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY orchestra ./orchestra

# Run unprivileged: the app takes arbitrary text from the network and fans it
# out to third-party APIs, so there is no reason for it to be root.
RUN useradd --create-home --uid 10001 council && chown -R council:council /app
USER council

EXPOSE 8000

# PaaS platforms inject their own $PORT; honour it rather than hardcoding.
CMD ["sh", "-c", "exec uvicorn orchestra.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
