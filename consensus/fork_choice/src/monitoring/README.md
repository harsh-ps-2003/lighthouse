# Lighthouse FCR Monitoring (Research Prototype)

Simple monitoring setup for Fast Confirmation Rule research, inspired by [Prysm's approach](https://ethresear.ch/t/fast-confirmation-rule-on-safe-head-in-prysm/22167).

## Quick Start

1. **Start monitoring:**
   ```bash
   make start
   ```

2. **Access dashboards:**
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## Prerequisites

- Prometheus and Grafana installed
- Lighthouse running with FCR enabled:
  ```bash
  ./target/release/lighthouse beacon_node \
    --network sepolia \
    --execution-endpoint http://127.0.0.1:8551 \
    --execution-jwt ~/.lighthouse-sepolia/jwt.hex \
    --datadir ~/.lighthouse-sepolia \
    --fast-confirmation \
    --fcr-byzantine-threshold 25 \
    --metrics --metrics-address 127.0.0.1 --metrics-port 5054
  ```

## What You'll See

- FCR safe head vs current head comparison
- Confirmation times and validator support
- Basic performance metrics

## Files

- `prometheus/prometheus.yml` - Prometheus config
- `grafana/grafana.ini` - Grafana config  
- `grafana/dashboards/fcr-overview.json` - FCR dashboard
- `scripts/start.sh` - Start monitoring
