FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY fixtures ./fixtures
COPY schemas ./schemas
COPY docs ./docs

RUN uv pip install --system -e ".[dev]"

ENV ELEANITY_RUNS_DIR=/data/runs
VOLUME ["/data"]

ENTRYPOINT ["eleanity"]
CMD ["doctor"]
