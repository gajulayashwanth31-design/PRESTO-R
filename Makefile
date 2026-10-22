.PHONY: test lint fmt quick metrics tables success figures report pdf package-models reproduce-results

test:
	python -m pytest

lint:
	ruff check presto tests scripts

fmt:
	black presto tests scripts

quick: test
	python -m scripts.smoke_env

metrics:
	python -m presto.eval.metrics --raw-dir results/raw --out results/episode_metrics.parquet

tables: metrics
	python -m presto.eval.stats --metrics results/episode_metrics.parquet --out-dir results/tables --horizon-predictor 3=results/predictor/horizon_H3/metrics.csv --horizon-predictor 10=results/predictor/horizon_H10/metrics.csv

success: tables
	python -m presto.eval.check_success

figures: tables
	python -m presto.eval.plots

report: tables
	python -m presto.eval.report
	python -m scripts.export_report_pdf

pdf:
	python -m scripts.export_report_pdf

package-models:
	python -m scripts.package_models

reproduce-results: success figures report
