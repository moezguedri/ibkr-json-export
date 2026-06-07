FROM python:3.12-slim

WORKDIR /app

# Install deps separately for layer-cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Output dirs are mounted as volumes at runtime; create them so the app can
# write without errors when no volume is attached.
RUN mkdir -p snapshots decisions

# Default: offline decision engine (no IBKR connection needed).
# Override with: docker run ... python ibkr_to_json.py --light
CMD ["python", "monthly_engine.py"]
