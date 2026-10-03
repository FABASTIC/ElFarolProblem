FROM node:22-bookworm-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    ELFAROL_PYTHON=/opt/venv/bin/python \
    ELFAROL_TORCH_HOST=native \
    ELFAROL_ALLOWED_HOSTS=* \
    PORT=4173 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch \
    && pip install --no-cache-dir -r requirements.txt

COPY dashboard/package.json dashboard/package-lock.json dashboard/
RUN npm --prefix dashboard ci

COPY . .
RUN npm --prefix dashboard run build

EXPOSE 4173
VOLUME ["/app/outputs"]

CMD ["sh", "-c", "mkdir -p outputs/experiment && { [ -n \"$(ls -A outputs/experiment)\" ] || cp -r dashboard/public/data/. outputs/experiment/; } && cd dashboard && exec npx vite preview --host 0.0.0.0"]
