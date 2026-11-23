# MQTT Bridge for Isfahan UTC/SCADA Integration
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY mqtt_bridge.py .
COPY configs/ /app/configs/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import paho.mqtt.client as mqtt; c=mqtt.Client(); c.connect('localhost', 1883, 5); c.disconnect()" || exit 1

# Run
CMD ["python3", "mqtt_bridge.py"]