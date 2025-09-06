#!/bin/bash

# Simple start script for FCR monitoring (research prototype)

set -e

echo "Starting Lighthouse FCR monitoring for Sepolia testnet..."

# Create data directories
mkdir -p prometheus/data

# Start Prometheus
echo "Starting Prometheus..."
prometheus \
    --config.file=prometheus/prometheus.yml \
    --storage.tsdb.path=prometheus/data \
    --web.listen-address=0.0.0.0:9090 \
    --web.enable-lifecycle \
    > prometheus/prometheus.log 2>&1 &
echo $! > prometheus/prometheus.pid

echo "Monitoring started!"
echo "Prometheus: http://localhost:9090"
echo ""
echo "Make sure Lighthouse is running with FCR enabled:"
echo "  --fast-confirmation --fcr-byzantine-threshold 25 --metrics --metrics-address 127.0.0.1 --metrics-port 5054"
echo ""
echo "View FCR metrics at: http://localhost:9090/targets"
echo "Query FCR metrics: http://localhost:9090/graph"
