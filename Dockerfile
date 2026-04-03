# Dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir maturin
RUN pip install --no-cache-dir -r requirements.txt

# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY . .

RUN cd src/tournament/tcg_engine && maturin build --release
RUN pip install src/tournament/tcg_engine/target/wheels/*.whl

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]