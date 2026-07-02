# Build context is the repository root (see docker-compose.yml) so the image
# installs the local source rather than a possibly-stale PyPI release.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY vectormorph ./vectormorph
RUN pip install --no-cache-dir .

EXPOSE 4440

CMD ["vectormorph"]
