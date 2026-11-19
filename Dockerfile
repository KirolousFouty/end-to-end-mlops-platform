FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY api ./api
# The serving image intentionally installs only serving dependencies. Training,
# Prefect and Evidently run from the development/worker environment.
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi "uvicorn[standard]" pydantic sqlalchemy "psycopg[binary]" prometheus-client \
    && pip install --no-cache-dir --no-deps .
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
