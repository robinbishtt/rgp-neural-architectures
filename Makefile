# =============================================================================
# RGP Neural Architectures — Makefile
# =============================================================================

PYTHON      := python3
PYTEST      := pytest
PIP         := pip
PROJECT     := rgp_neural_architectures

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo "RGP Neural Architectures — Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup              Install all dependencies"
	@echo "    make check_env          Validate environment"
	@echo ""
	@echo "  Verification (Reviewer):"
	@echo "    make verify_pipeline    Smoke test < 1 min"
	@echo "    make reproduce_fast     Full fast-track 3-5 min"
	@echo "    make reproduce_fast_h1  H1 fast-track only"
	@echo "    make reproduce_fast_h2  H2 fast-track only"
	@echo "    make reproduce_fast_h3  H3 fast-track only"
	@echo ""
	@echo "  Full Experiments:"
	@echo "    make reproduce_all      Complete pipeline"
	@echo "    make reproduce_h1       H1 validation"
	@echo "    make reproduce_h2       H2 validation"
	@echo "    make reproduce_h3       H3 validation"
	@echo ""
	@echo "  Figures:"
	@echo "    make reproduce_figures           All figures"
	@echo "    make reproduce_extended_data     Extended Data figures"
	@echo "    make reproduce_tables            All tables"
	@echo ""
	@echo "  Testing:"
	@echo "    make test               All tests"
	@echo "    make test_unit          Unit tests only"
	@echo "    make test_integration   Integration tests"
	@echo "    make test_stability     Stability tests"
	@echo "    make test_spectral      Spectral / RMT tests"
	@echo "    make validate           Full validation suite"
	@echo ""
	@echo "  Audit (structured JSON trail):"
	@echo "    make audit                   Full audit - all suites + study scripts"
	@echo "    make audit_tests             All 8 pytest suites only"
	@echo "    make audit_studies           All 5 study/figure suites only"
	@echo "    make audit_unit              tests/unit only"
	@echo "    make audit_integration       tests/integration only"
	@echo "    make audit_stability         tests/stability only"
	@echo "    make audit_ablation          tests/ablation only"
	@echo "    make audit_robustness        tests/robustness only"
	@echo "    make audit_scaling           tests/scaling only"
	@echo "    make audit_spectral          tests/spectral only"
	@echo "    make audit_validation        tests/validation only"
	@echo "    make audit_h1                H1 experiment scripts only"
	@echo "    make audit_h2                H2 experiment scripts only"
	@echo "    make audit_h3                H3 experiment scripts only"
	@echo "    make audit_figures_extended  Extended Data figures only"
	@echo "    make audit_figures_manuscript Manuscript figures only"
	@echo ""
	@echo "  Code quality:"
	@echo "    make lint               flake8 + isort check"
	@echo "    make format             black + isort format"
	@echo "    make typecheck          mypy type check"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean              Remove generated artifacts"
	@echo "    make clean_all          Remove all outputs"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: setup
setup:
	$(PIP) install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
		--index-url https://download.pytorch.org/whl/cu118 || \
		$(PIP) install torch==2.0.1+cpu torchvision==0.15.2+cpu \
		--index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	@echo "Setup complete."

.PHONY: check_env
check_env:
	$(PYTHON) scripts/verify_pipeline.py --check-env

# ---------------------------------------------------------------------------
# Reviewer verification
# ---------------------------------------------------------------------------
.PHONY: verify_pipeline
verify_pipeline:
	$(PYTHON) scripts/verify_pipeline.py

.PHONY: reproduce_fast
reproduce_fast:
	bash scripts/reproduce_fast.sh

.PHONY: reproduce_fast_h1
reproduce_fast_h1:
	bash scripts/reproduce_fast.sh --hypothesis h1

.PHONY: reproduce_fast_h2
reproduce_fast_h2:
	bash scripts/reproduce_fast.sh --hypothesis h2

.PHONY: reproduce_fast_h3
reproduce_fast_h3:
	bash scripts/reproduce_fast.sh --hypothesis h3

# ---------------------------------------------------------------------------
# Full experiments
# ---------------------------------------------------------------------------
.PHONY: reproduce_all
reproduce_all: reproduce_h1 reproduce_h2 reproduce_h3 reproduce_figures reproduce_tables
	@echo "Full reproduction complete."

.PHONY: reproduce_h1
reproduce_h1:
	$(PYTHON) experiments/h1_scale_correspondence/run_h1_validation.py

.PHONY: reproduce_h2
reproduce_h2:
	$(PYTHON) experiments/h2_depth_scaling/run_h2_validation.py

.PHONY: reproduce_h3
reproduce_h3:
	$(PYTHON) experiments/h3_multiscale_generalization/run_h3_validation.py

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
.PHONY: reproduce_figures
reproduce_figures:
	$(PYTHON) figures/generate_all.py --results-root results/ --output figures/out/

.PHONY: reproduce_fast_figures
reproduce_fast_figures:
	$(PYTHON) figures/generate_all.py --fast-track --output figures/out/

.PHONY: reproduce_extended_data
reproduce_extended_data:
	$(PYTHON) figures/generate_all.py \
		--figures ed_fig1 ed_fig2 ed_fig3 ed_fig4 ed_fig5 ed_fig6 \
		--results-root results/ --output figures/out/

.PHONY: reproduce_tables
reproduce_tables:
	bash scripts/reproduce_tables.sh

.PHONY: reproduce_figures_from_checkpoints
reproduce_figures_from_checkpoints:
	$(PYTHON) scripts/extract_from_checkpoints.py
	$(PYTHON) figures/generate_all.py --results-root results/ --output figures/out/

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
.PHONY: test
test:
	$(PYTEST) tests/ -v --timeout=300

.PHONY: test_unit
test_unit:
	$(PYTEST) tests/unit/ -v

.PHONY: test_integration
test_integration:
	$(PYTEST) tests/integration/ -v --timeout=600

.PHONY: test_stability
test_stability:
	$(PYTEST) tests/stability/ -v

.PHONY: test_spectral
test_spectral:
	$(PYTEST) tests/spectral/ -v

.PHONY: test_scaling
test_scaling:
	$(PYTEST) tests/scaling/ -v

.PHONY: validate
validate:
	bash scripts/validate_determinism.sh
	bash scripts/validate_hypotheses.sh

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
.PHONY: lint
lint:
	flake8 src/ tests/ experiments/ figures/ --max-line-length=100 --ignore=E501,W503
	isort --check-only src/ tests/

.PHONY: format
format:
	black src/ tests/ experiments/ figures/ --line-length=100
	isort src/ tests/

.PHONY: typecheck
typecheck:
	mypy src/ --ignore-missing-imports

# ---------------------------------------------------------------------------
# Audit — structured JSON trail
# ---------------------------------------------------------------------------
# Full audit (all pytest suites + all study/figure scripts):
#   make audit
#
# Tests only (8 pytest suites, no study scripts):
#   make audit_tests
#
# Study/figure scripts only (no pytest):
#   make audit_studies
#
# Per-suite (pytest):
#   make audit_unit | audit_integration | audit_stability | audit_ablation
#   make audit_robustness | audit_scaling | audit_spectral | audit_validation
#
# Per-suite (study):
#   make audit_h1 | audit_h2 | audit_h3
#   make audit_figures_extended | audit_figures_manuscript

.PHONY: audit
audit:
	$(PYTHON) audit/run_audit.py --seed 42

.PHONY: audit_tests
audit_tests:
	$(PYTHON) audit/run_audit.py --tests-only --seed 42

.PHONY: audit_studies
audit_studies:
	$(PYTHON) audit/run_audit.py --studies-only --seed 42

# --- Per pytest suite -------------------------------------------------------
.PHONY: audit_unit
audit_unit:
	$(PYTHON) audit/run_suite.py unit --seed 42

.PHONY: audit_integration
audit_integration:
	$(PYTHON) audit/run_suite.py integration --seed 42

.PHONY: audit_stability
audit_stability:
	$(PYTHON) audit/run_suite.py stability --seed 42

.PHONY: audit_ablation
audit_ablation:
	$(PYTHON) audit/run_suite.py ablation --seed 42

.PHONY: audit_robustness
audit_robustness:
	$(PYTHON) audit/run_suite.py robustness --seed 42

.PHONY: audit_scaling
audit_scaling:
	$(PYTHON) audit/run_suite.py scaling --seed 42

.PHONY: audit_spectral
audit_spectral:
	$(PYTHON) audit/run_suite.py spectral --seed 42

.PHONY: audit_validation
audit_validation:
	$(PYTHON) audit/run_suite.py validation --seed 42

# --- Per study suite --------------------------------------------------------
.PHONY: audit_h1
audit_h1:
	$(PYTHON) audit/run_suite.py h1_scale_correspondence --seed 42

.PHONY: audit_h2
audit_h2:
	$(PYTHON) audit/run_suite.py h2_depth_scaling --seed 42

.PHONY: audit_h3
audit_h3:
	$(PYTHON) audit/run_suite.py h3_multiscale_generalization --seed 42

.PHONY: audit_figures_extended
audit_figures_extended:
	$(PYTHON) audit/run_suite.py figures_extended_data --seed 42

.PHONY: audit_figures_manuscript
audit_figures_manuscript:
	$(PYTHON) audit/run_suite.py figures_manuscript --seed 42

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/

.PHONY: clean_all
clean_all: clean
	rm -rf results/ figures/out/ checkpoints/ logs/
	@echo "All generated artifacts removed."
