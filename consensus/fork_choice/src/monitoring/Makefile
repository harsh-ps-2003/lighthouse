# Lighthouse FCR monitoring 

.PHONY: help start stop

help:
	@echo "Lighthouse FCR Monitoring (Research Prototype)"
	@echo ""
	@echo "Commands:"
	@echo "  start   - Start Prometheus"
	@echo "  stop    - Stop Prometheus"
	@echo ""
	@echo "Access:"
	@echo "  Prometheus: http://localhost:9090"

start:
	@echo "Starting Prometheus..."
	@./scripts/start.sh

stop:
	@echo "Stopping Prometheus..."
	@if [ -f prometheus/prometheus.pid ]; then \
		kill $$(cat prometheus/prometheus.pid) 2>/dev/null || true; \
		rm -f prometheus/prometheus.pid; \
		echo "Prometheus stopped"; \
	else \
		echo "Prometheus not running"; \
	fi