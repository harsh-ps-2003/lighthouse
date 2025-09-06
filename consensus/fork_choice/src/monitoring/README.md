# Lighthouse FCR Monitoring (Research Prototype)

Simple Prometheus monitoring setup for Fast Confirmation Rule research, inspired by [Prysm's approach](https://ethresear.ch/t/fast-confirmation-rule-on-safe-head-in-prysm/22167).

## Quick Start

1. **Start monitoring:**
   ```bash
   make start
   ```

2. **Access Prometheus:**
   - Prometheus: http://localhost:9090
   - Query FCR metrics: http://localhost:9090/graph
   - Check targets: http://localhost:9090/targets

3. **Stop monitoring:**
   ```bash
   make stop
   ```

## Prerequisites

- Prometheus installed
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

## Key FCR Metrics

- `fcr_confirmation_time_seconds` - Time from block creation to confirmation (target: 12-24s)
- `fcr_slot_confirmation_delay` - Slots between block and confirmation (target: 1-2 slots)
- `fcr_time_compliance_ratio` - Percentage of confirmations within 12-24s window
- `fcr_slot_compliance_ratio` - Percentage of confirmations within 1-2 slot window
- `fcr_avg_confirmation_time_5m` - 5-minute rolling average confirmation time
- `fcr_p95_confirmation_time_5m` - 95th percentile confirmation time

## Files

- `prometheus/prometheus.yml` - Prometheus config
- `prometheus/fcr_rules.yml` - FCR-specific recording rules and alerts
- `scripts/start.sh` - Start monitoring script
