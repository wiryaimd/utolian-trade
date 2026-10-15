FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    STATE_DIR=/app/state

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[live]"

VOLUME /app/state
# Healthy while a closed 1m bar was processed in the last 10 minutes.
HEALTHCHECK --interval=2m --timeout=10s --start-period=15m \
  CMD python -c "import os,sys,time; p='/app/state/heartbeat'; sys.exit(0 if os.path.exists(p) and time.time()*1000-int(open(p).read())<600000 else 1)"

CMD ["python", "-m", "btcq.live", "run"]
