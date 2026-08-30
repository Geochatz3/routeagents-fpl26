# Makefile for FPGA Design Optimization Agent

# Configuration
PYTHON := python3
PIP := $(PYTHON) -m pip

# Optional .env at the repo root: simple KEY=value lines (no spaces around
# '=', no quotes needed). Loaded for every make target; the API key is
# exported to recipe shells. Invoking dcp_optimizer.py directly bypasses
# this — export the variables yourself or pass --api-key.
-include .env
# Everything .env can legitimately carry must reach the Python process,
# not just the recipe shell's make variables: the README variables table
# and .env.example both promise that a value set in .env takes effect.
export OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENROUTER_MODEL
export FPL26_RUN_DIR_BASE STRATEGY_MEMORY_PATH FPL26_WSL2_WIN_CWD

# Vivado executable - can be overridden with: make setup VIVADO_EXEC=/path/to/vivado
VIVADO_EXEC ?= vivado
export VIVADO_EXEC

# Set JAVA_HOME from PATH or Vivado if not already set
# Python RapidWright may need JAVA_HOME to be set, but often users only have `java` on PATH
# See: https://www.rapidwright.io/docs/Install.html#using-java-distributed-with-vivado
ifndef JAVA_HOME
  # ORDER MATTERS: prefer Vivado's bundled JRE over a PATH java.
  # The only validated eval-box configuration is JAVA_HOME = Vivado's
  # jre11 (a wrong system java makes RapidWright fail with a FAKE
  # structural failure). A preset JAVA_HOME still wins via the ifndef.
  # Vivado includes Java at: <VIVADO_ROOT>/tps/lnx64/jre*/bin/java.
  # Search for jre11 first (LTS, well-tested with RapidWright), fall back
  # to jre21 or any newer bundled JRE.  Vivado 2025.1 ships both jre11 and
  # jre21; older Vivado versions only ship jre11.
  VIVADO_PATH := $(shell command -v $(VIVADO_EXEC) 2>/dev/null)
  ifneq ($(VIVADO_PATH),)
    VIVADO_ROOT := $(shell dirname $(shell dirname $(VIVADO_PATH)))
    VIVADO_JAVA := $(shell ls $(VIVADO_ROOT)/tps/lnx64/jre11*/bin/java 2>/dev/null | head -n 1)
    ifeq ($(VIVADO_JAVA),)
      VIVADO_JAVA := $(shell ls $(VIVADO_ROOT)/tps/lnx64/jre*/bin/java 2>/dev/null | head -n 1)
    endif
  endif
  ifneq ($(VIVADO_JAVA),)
    export JAVA_HOME := $(shell dirname $(shell dirname $(VIVADO_JAVA)))
    export PATH := $(JAVA_HOME)/bin:$(PATH)
  else
    # No Vivado on PATH - fall back to whatever java is on PATH.
    # Resolve symlinks; try readlink -f (Linux), else direct path (macOS).
    JAVA_PATH := $(shell command -v java 2>/dev/null)
    ifneq ($(JAVA_PATH),)
      REAL_JAVA_PATH := $(shell readlink -f "$(JAVA_PATH)" 2>/dev/null || readlink "$(JAVA_PATH)" 2>/dev/null || echo "$(JAVA_PATH)")
      # java is at $JAVA_HOME/bin/java, so go up two directories
      export JAVA_HOME := $(shell dirname $(shell dirname $(REAL_JAVA_PATH)))
    endif
  endif
endif

# RapidWright submodule path and classpath
# Points the Python rapidwright package to use the local RapidWright source.
# See: https://www.rapidwright.io/docs/Install_RapidWright_as_a_Python_PIP_Package.html#java-development-and-python
#
# Only export RAPIDWRIGHT_PATH/CLASSPATH if a built submodule is present
# (build/libs/rapidwright.jar exists).  When the repo is unpacked from a
# release archive without the submodule, exporting an empty
# RAPIDWRIGHT_PATH/CLASSPATH causes pip rapidwright's start_jvm() to skip
# its own bundled jar (it gates on `if not RAPIDWRIGHT_PATH`), the JVM
# starts with an empty classpath, and `from com.xilinx... import ...`
# fails with `Failed to import 'com.xilinx'` — exactly the
# JVMNotFoundException-look-alike we want to avoid.
RAPIDWRIGHT_PATH := $(CURDIR)/RapidWright
ifneq ($(wildcard $(RAPIDWRIGHT_PATH)/build/libs/rapidwright.jar),)
  export RAPIDWRIGHT_PATH
  export CLASSPATH := $(RAPIDWRIGHT_PATH)/bin:$(RAPIDWRIGHT_PATH)/jars/*:$(RAPIDWRIGHT_PATH)/build/libs/*
else
  # No built submodule.  Leave RAPIDWRIGHT_PATH and CLASSPATH unset so the
  # pip-installed rapidwright uses its own bundled standalone jar.
endif

# Benchmark archive from GitHub release
BENCHMARK_VERSION := v1.2.0
BENCHMARK_TARBALL := fpl26_contest_benchmarks_$(BENCHMARK_VERSION).tar.gz
BENCHMARK_DIR := fpl26_contest_benchmarks
BENCHMARK_URL := https://github.com/Xilinx/fpl26_optimization_contest/releases/download/$(BENCHMARK_VERSION)/$(BENCHMARK_TARBALL)
# Version marker written into the extracted benchmark directory so `make setup`
# can detect when an older archive was extracted previously and needs to be
# refreshed. Without this, `make setup` would silently keep a stale
# fpl26_contest_benchmarks/ from a previous BENCHMARK_VERSION.
BENCHMARK_VERSION_FILE := $(BENCHMARK_DIR)/.benchmark_version

# Example DCP paths (inside the extracted benchmark directory)
EXAMPLE_DCP_1 := $(BENCHMARK_DIR)/logicnets_jscl_2025.1.dcp
EXAMPLE_DCP_2 := $(BENCHMARK_DIR)/vexriscv_re-place_2025.1.dcp

# Colors for output
COLOR_GREEN := \033[0;32m
COLOR_YELLOW := \033[0;33m
COLOR_RED := \033[0;31m
COLOR_BLUE := \033[0;34m
COLOR_RESET := \033[0m

.PHONY: setup download_dcps build-rapidwright run_optimizer run_no_llm test \
        lint validate validate_demo submission \
        clean veryclean help

# Default target
help:
	@echo "FPGA Design Optimization Agent - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup              - Install dependencies, build RapidWright, download example DCPs"
	@echo "  build-rapidwright  - Build RapidWright from source (if a local clone is present)"
	@echo "  run_optimizer      - Run optimizer on a DCP file (LLM-guided, requires API key)"
	@echo "  run_no_llm         - Run the optimizer with the LLM disabled (deterministic path only)"
	@echo "  run_once           - One optimization attempt, no flag injection (run_optimizer calls this per attempt)"
	@echo "  test               - Run the offline test suite (no Vivado / LLM / network)"
	@echo "  validate           - Validate functional equivalence between two DCPs"
	@echo "  validate_demo      - Run validation demo (self-check)"
	@echo "  run_optimizer_multirestart - Best-of-N: repeat, keep the best validated attempt"
	@echo "  download_dcps      - Fetch the example benchmark DCPs"
	@echo "  submission         - Package this tree as a contest-format tarball"
	@echo "  clean              - Remove generated files (run directories, logs, Vivado outputs)"
	@echo "  veryclean          - Remove all generated files including example DCPs"
	@echo ""
	@echo "Usage examples:"
	@echo "  make setup"
	@echo "  make setup VIVADO_EXEC=/tools/Xilinx/Vivado/2025.2/bin/vivado"
	@echo "  make run_optimizer DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp"
	@echo "  make run_no_llm DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp"
	@echo "  make run_no_llm DCP=fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp"
	@echo "  make validate GOLDEN=design.dcp REVISED=design_optimized.dcp"
	@echo "  make validate GOLDEN=design.dcp REVISED=design_optimized.dcp VECTORS=50000"
	@echo "  make validate_demo"
	@echo "  make clean"
	@echo ""
	@echo "Environment variables:"
	@echo "  VIVADO_EXEC     - Path to Vivado executable (default: vivado)"
	@echo "  JAVA_HOME       - Java installation directory (auto-detected from PATH if not set)"
	@echo "  DCP             - Input DCP file for run_optimizer / run_no_llm targets"
	@echo "  MAX_NETS        - Max high fanout nets to optimize in test mode (default: 5)"
	@echo "  GOLDEN          - Golden (reference) DCP for validation"
	@echo "  REVISED         - Revised (optimized) DCP for validation"
	@echo "  VECTORS         - Number of test vectors for validation (default: 200)"
	@echo ""
	@echo "Output structure:"
	@echo "  - Optimized DCP: <input_name>_optimized-<timestamp>.dcp (next to input)"
	@echo "  - Run directory: dcp_optimizer_run-<timestamp>/ (contains all logs)"
	@echo "  - Validation:    /tmp/dcp_validation_*/ (contains simulation logs)"

# Setup target: Install dependencies, check Vivado, set up Java, build RapidWright, download DCPs
setup:
	@printf "$(COLOR_GREEN)===== FPGA Design Optimization Setup =====$(COLOR_RESET)\n"
	@echo ""
	
	@printf "$(COLOR_YELLOW)[1/5] Installing Python dependencies...$(COLOR_RESET)\n"
	$(PIP) install -r requirements.txt
	@printf "$(COLOR_GREEN)✓ Python dependencies installed$(COLOR_RESET)\n"
	@echo ""
	
	@printf "$(COLOR_YELLOW)[2/5] Checking Vivado...$(COLOR_RESET)\n"
	@if command -v $(VIVADO_EXEC) >/dev/null 2>&1; then \
		printf "$(COLOR_GREEN)✓ Vivado found: %s$(COLOR_RESET)\n" "$$(command -v $(VIVADO_EXEC))"; \
		$(VIVADO_EXEC) -version | head -n 1; \
	else \
		printf "$(COLOR_RED)✗ Vivado not found on PATH$(COLOR_RESET)\n"; \
		echo ""; \
		echo "Please either:"; \
		echo "  1. Source Vivado settings: source /path/to/Vivado/*/settings64.sh"; \
		echo "  2. Specify Vivado path: make setup VIVADO_EXEC=/path/to/vivado"; \
		exit 1; \
	fi
	@echo ""
	
	@printf "$(COLOR_YELLOW)[3/5] Checking Java...$(COLOR_RESET)\n"
	@if command -v java >/dev/null 2>&1; then \
		printf "$(COLOR_GREEN)✓ Java found: %s$(COLOR_RESET)\n" "$$(command -v java)"; \
		java -version 2>&1 | head -n 1; \
	else \
		printf "$(COLOR_YELLOW)⚠ Java not found on PATH$(COLOR_RESET)\n"; \
		echo "Attempting to locate Java from Vivado installation..."; \
		VIVADO_PATH=$$(command -v $(VIVADO_EXEC)); \
		if [ -n "$$VIVADO_PATH" ]; then \
			VIVADO_BIN_DIR=$$(dirname $$VIVADO_PATH); \
			VIVADO_ROOT=$$(dirname $$VIVADO_BIN_DIR); \
			JAVA_FOUND=$$(ls $$VIVADO_ROOT/tps/lnx64/jre11*/bin/java 2>/dev/null | head -n 1); \
			if [ -z "$$JAVA_FOUND" ]; then \
				JAVA_FOUND=$$(ls $$VIVADO_ROOT/tps/lnx64/jre*/bin/java 2>/dev/null | head -n 1); \
			fi; \
			if [ -n "$$JAVA_FOUND" ]; then \
				printf "$(COLOR_GREEN)✓ Found Java in Vivado: %s$(COLOR_RESET)\n" "$$JAVA_FOUND"; \
				echo ""; \
				printf "$(COLOR_YELLOW)NOTE: Set JAVA_HOME before running optimizer:$(COLOR_RESET)\n"; \
				JAVA_HOME_DIR=$$(dirname $$(dirname $$JAVA_FOUND)); \
				echo "  export JAVA_HOME=$$JAVA_HOME_DIR"; \
				echo "  export PATH=\$$JAVA_HOME/bin:\$$PATH"; \
			else \
				printf "$(COLOR_RED)✗ Could not find Java in Vivado installation$(COLOR_RESET)\n"; \
				echo "Please install Java 11 or later"; \
				exit 1; \
			fi; \
		else \
			printf "$(COLOR_RED)✗ Cannot locate Java$(COLOR_RESET)\n"; \
			echo "Please install Java 11 or later"; \
			exit 1; \
		fi; \
	fi
	@echo ""
	
	@printf "$(COLOR_YELLOW)[4/5] Building RapidWright from source...$(COLOR_RESET)\n"
	@$(MAKE) build-rapidwright
	@echo ""

	@printf "$(COLOR_YELLOW)[4.5/5] Pre-warming RapidWright xcvu3p device data...$(COLOR_RESET)\n"
	@# RapidWright downloads device data files ON FIRST USE. The contest
	@# forbids any network access during the scored run (only OpenRouter),
	@# so force the xcvu3p (contest device) download NOW, during setup,
	@# where network access is part of the sanctioned install flow.
	@# Non-fatal: a failure here only means RapidWright tools may be
	@# unavailable at runtime (the agent already tolerates that).
	@$(PYTHON) -c "import rapidwright; from com.xilinx.rapidwright.device import Device; d = Device.getDevice('xcvu3p'); print('  prewarmed:', d.getName())" \
		&& printf "$(COLOR_GREEN)✓ xcvu3p device data cached$(COLOR_RESET)\n" \
		|| printf "$(COLOR_YELLOW)⚠ RapidWright device prewarm failed (RapidWright tools may be degraded at runtime)$(COLOR_RESET)\n"
	@echo ""

	@printf "$(COLOR_YELLOW)[5/5] Downloading and extracting benchmark DCPs...$(COLOR_RESET)\n"
	@$(MAKE) --no-print-directory download_dcps
	@echo ""
	
	@printf "$(COLOR_GREEN)===== Setup Complete! =====$(COLOR_RESET)\n"
	@echo ""
	@echo "Next steps - run the optimizer:"
	@echo ""
	@echo "  Test mode (no API key required):"
	@echo "    make run_no_llm DCP=$(EXAMPLE_DCP_1)"
	@echo "    make run_no_llm DCP=$(EXAMPLE_DCP_2)"
	@echo ""
	@echo "  Full LLM-guided optimizer (requires OPENROUTER_API_KEY):"
	@echo "    make run_optimizer DCP=$(EXAMPLE_DCP_1)"
	@echo ""
	@echo "Output will be in:"
	@echo "  - Optimized DCP: <input_name>_optimized-<timestamp>.dcp"
	@echo "  - Run logs: dcp_optimizer_run-<timestamp>/"
	@echo ""

# Download and extract the benchmark DCPs into $(BENCHMARK_DIR)/.
# Called by `setup`, and directly by `validate_demo` when the example
# DCP it needs is missing. Idempotent: an existing directory at the
# current BENCHMARK_VERSION is left alone.
download_dcps:
	@if [ -d "$(BENCHMARK_DIR)" ] && [ -f "$(EXAMPLE_DCP_1)" ] && \
	    [ -f "$(BENCHMARK_VERSION_FILE)" ] && \
	    [ "$$(cat $(BENCHMARK_VERSION_FILE))" = "$(BENCHMARK_VERSION)" ]; then \
		printf "$(COLOR_GREEN)✓ $(BENCHMARK_DIR)/ already exists for $(BENCHMARK_VERSION)$(COLOR_RESET)\n"; \
	else \
		if [ -d "$(BENCHMARK_DIR)" ]; then \
			if [ -f "$(BENCHMARK_VERSION_FILE)" ]; then \
				EXISTING_VERSION=$$(cat "$(BENCHMARK_VERSION_FILE)"); \
				printf "$(COLOR_YELLOW)Existing $(BENCHMARK_DIR)/ is version %s; refreshing to $(BENCHMARK_VERSION)...$(COLOR_RESET)\n" "$$EXISTING_VERSION"; \
			else \
				printf "$(COLOR_YELLOW)Existing $(BENCHMARK_DIR)/ has no version marker; refreshing to $(BENCHMARK_VERSION)...$(COLOR_RESET)\n"; \
			fi; \
			rm -rf "$(BENCHMARK_DIR)"; \
		fi; \
		if [ ! -f "$(BENCHMARK_TARBALL)" ]; then \
			printf "Downloading $(BENCHMARK_TARBALL)...\n"; \
			if command -v wget >/dev/null 2>&1; then \
				wget -q --show-progress $(BENCHMARK_URL); \
			elif command -v curl >/dev/null 2>&1; then \
				curl -# -L -O $(BENCHMARK_URL); \
			else \
				printf "$(COLOR_RED)✗ Neither wget nor curl found$(COLOR_RESET)\n"; \
				echo "Please install wget or curl, or manually download:"; \
				echo "  $(BENCHMARK_URL)"; \
				exit 1; \
			fi; \
		fi; \
		printf "Extracting $(BENCHMARK_TARBALL)...\n"; \
		tar xzf $(BENCHMARK_TARBALL); \
		echo "$(BENCHMARK_VERSION)" > "$(BENCHMARK_VERSION_FILE)"; \
		printf "$(COLOR_GREEN)✓ Benchmarks extracted to $(BENCHMARK_DIR)/ ($(BENCHMARK_VERSION))$(COLOR_RESET)\n"; \
	fi
	@echo ""

# Build RapidWright from source (optional local clone at ./RapidWright).
#
# The pip-installed `rapidwright` package ships its own bundled jars and is
# what RapidWrightMCP actually imports at runtime, so the source build is
# OPTIONAL: it only happens when a RapidWright clone is already present at
# $(RAPIDWRIGHT_PATH) (e.g. `git clone https://github.com/Xilinx/RapidWright`).
# In every other case this target is a no-op and pip rapidwright is used.
#
# Recipe is a single shell so early-exit propagates correctly.
build-rapidwright:
	@set -e; \
	printf "$(COLOR_YELLOW)Building RapidWright from source...$(COLOR_RESET)\n"; \
	if [ ! -f "$(RAPIDWRIGHT_PATH)/gradlew" ]; then \
		printf "$(COLOR_YELLOW)⚠ No local RapidWright clone at $(RAPIDWRIGHT_PATH) — skipping source build.$(COLOR_RESET)\n"; \
		printf "$(COLOR_YELLOW)  pip-installed rapidwright will be used at runtime.$(COLOR_RESET)\n"; \
		exit 0; \
	fi; \
	cd "$(RAPIDWRIGHT_PATH)" && ./gradlew compileJava -p "$(RAPIDWRIGHT_PATH)"; \
	printf "$(COLOR_GREEN)✓ RapidWright built successfully$(COLOR_RESET)\n"; \
	printf "$(COLOR_GREEN)  RAPIDWRIGHT_PATH=$(RAPIDWRIGHT_PATH)$(COLOR_RESET)\n"; \
	printf "$(COLOR_GREEN)  CLASSPATH=$(CLASSPATH)$(COLOR_RESET)\n"

# Run optimizer target: Run dcp_optimizer.py (output DCP name generated automatically)
run_optimizer: 
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_optimizer DCP=input.dcp"; \
		exit 1; \
	fi
	@if [ ! -f "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP file not found: $(DCP)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_GREEN)Running optimizer on $(DCP)...$(COLOR_RESET)\n"
	@# Set up Java from Vivado if Java is not available.  Search jre11 first
	@# (LTS, primary RapidWright target), then any newer bundled JRE (jre21
	@# in Vivado 2025.1, future-proofing for later releases).
	@if ! command -v java >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)Java not found on PATH, attempting to use Java from Vivado...$(COLOR_RESET)\n"; \
		VIVADO_PATH=$$(command -v $(VIVADO_EXEC) 2>/dev/null); \
		if [ -n "$$VIVADO_PATH" ]; then \
			VIVADO_BIN_DIR=$$(dirname $$VIVADO_PATH); \
			VIVADO_ROOT=$$(dirname $$VIVADO_BIN_DIR); \
			JAVA_FOUND=$$(ls $$VIVADO_ROOT/tps/lnx64/jre11*/bin/java 2>/dev/null | head -n 1); \
			if [ -z "$$JAVA_FOUND" ]; then \
				JAVA_FOUND=$$(ls $$VIVADO_ROOT/tps/lnx64/jre*/bin/java 2>/dev/null | head -n 1); \
			fi; \
			if [ -n "$$JAVA_FOUND" ]; then \
				export JAVA_HOME=$$(dirname $$(dirname $$JAVA_FOUND)); \
				export PATH="$$JAVA_HOME/bin:$$PATH"; \
				printf "$(COLOR_GREEN)Using Java from Vivado: %s$(COLOR_RESET)\n" "$$JAVA_HOME"; \
			else \
				printf "$(COLOR_RED)Java not found in Vivado install at $$VIVADO_ROOT/tps/lnx64/jre*$(COLOR_RESET)\n"; \
				exit 1; \
			fi; \
		else \
			printf "$(COLOR_RED)Vivado not found on PATH; cannot derive JAVA_HOME$(COLOR_RESET)\n"; \
			exit 1; \
		fi; \
	fi; \
	echo ""; \
	$(if $(POLISH_RESERVE_S),FPL26_POLISH_RESERVE_S=$(POLISH_RESERVE_S)) FPL26_DEEP_WNS_TAIL_RESERVE=$(if $(DEEP_WNS_TAIL_RESERVE),$(DEEP_WNS_TAIL_RESERVE),2400) FPL26_DEEP_REPLACE=$(if $(DEEP_REPLACE),$(DEEP_REPLACE),1) FPL26_DEEP_REPLACE_FIRST=$(if $(DEEP_REPLACE_FIRST),$(DEEP_REPLACE_FIRST),1) FPL26_DEEP_REPLACE_UNBANDED=$(if $(DEEP_REPLACE_UNBANDED),$(DEEP_REPLACE_UNBANDED),1) FPL26_DEEP_FIRST_SIZEGATED=$(if $(DEEP_FIRST_SIZEGATED),$(DEEP_FIRST_SIZEGATED),1) FPL26_DEEP_REPLACE_B3=$(if $(DEEP_REPLACE_B3),$(DEEP_REPLACE_B3),1) FPL26_B3_FLOOR_EXIT=$(if $(B3_FLOOR_EXIT),$(B3_FLOOR_EXIT),1) FPL26_LOGIC_FLOOR_EXIT=$(if $(LOGIC_FLOOR_EXIT),$(LOGIC_FLOOR_EXIT),1) FPL26_ILS_HURDLE_CONTINUE=$(if $(ILS_HURDLE),$(ILS_HURDLE),1) FPL26_ILS_MEASURED_PRIORS=$(if $(MEASURED_PRIORS),$(MEASURED_PRIORS),1) FPL26_PHYSOPT_DEFAULT_FIXPOINT=$(if $(PODF),$(PODF),0) FPL26_ILS_LADDER_ORDER_BY_WNS=$(if $(LADDER_WNS),$(LADDER_WNS),1) FPL26_RECIPE_PASS=$(if $(RECIPE_PASS),$(RECIPE_PASS),1) FPL26_RECIPE_FIRST_DEEP=$(if $(RECIPE_FIRST_DEEP),$(RECIPE_FIRST_DEEP),1) FPL26_SUBBAND_PHYSOPT_FLOOR=$(if $(SUBBAND_PHYSOPT_FLOOR),$(SUBBAND_PHYSOPT_FLOOR),1) FPL26_ETO_RETIME_CANDIDATE=$(if $(ETO_RETIME),$(ETO_RETIME),1) FPL26_MIDBAND_RETRY_HOLD=$(if $(MIDBAND_RETRY_HOLD),$(MIDBAND_RETRY_HOLD),1) FPL26_MIDBAND_ROUTE_RUNG=$(if $(MIDBAND_ROUTE_RUNG),$(MIDBAND_ROUTE_RUNG),1) FPL26_OWNFRONT_RETIME_CANDIDATE=$(if $(OWNFRONT_RETIME),$(OWNFRONT_RETIME),1) FPL26_MUX_MD5_TRUST=$(if $(MUX_MD5_TRUST),$(MUX_MD5_TRUST),1) FPL26_SHALLOW_DETERMINIZER_CANDIDATE=$(if $(SHALLOW_DET),$(SHALLOW_DET),1) FPL26_POSITIVE_SLACK_CONTINUE=$(if $(POSITIVE_SLACK),$(POSITIVE_SLACK),1) $(PYTHON) scripts/multi_restart_optimize.py "$(DCP)" \
		--total-wall $(if $(MAX_WALL),$(MAX_WALL),3500) \
		--max-attempts $(if $(MAX_ATTEMPTS),$(MAX_ATTEMPTS),4) \
		--cost-cap $(if $(COST_CAP),$(COST_CAP),0.85) \
		--cost-ceiling $(if $(COST_CEILING),$(COST_CEILING),0.80) \
		$(if $(filter 0,$(ILS)),,--ils-polish) \
		$(if $(filter 1 true yes on,$(SPLIT_AWARE)),--split-aware) \
		$(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback) \
		|| FPL26_DEEP_WNS_TAIL_RESERVE=$(if $(DEEP_WNS_TAIL_RESERVE),$(DEEP_WNS_TAIL_RESERVE),2400) FPL26_DEEP_REPLACE=$(if $(DEEP_REPLACE),$(DEEP_REPLACE),1) FPL26_DEEP_REPLACE_FIRST=$(if $(DEEP_REPLACE_FIRST),$(DEEP_REPLACE_FIRST),1) FPL26_DEEP_REPLACE_UNBANDED=$(if $(DEEP_REPLACE_UNBANDED),$(DEEP_REPLACE_UNBANDED),1) FPL26_DEEP_FIRST_SIZEGATED=$(if $(DEEP_FIRST_SIZEGATED),$(DEEP_FIRST_SIZEGATED),1) FPL26_DEEP_REPLACE_B3=$(if $(DEEP_REPLACE_B3),$(DEEP_REPLACE_B3),1) FPL26_B3_FLOOR_EXIT=$(if $(B3_FLOOR_EXIT),$(B3_FLOOR_EXIT),1) FPL26_LOGIC_FLOOR_EXIT=$(if $(LOGIC_FLOOR_EXIT),$(LOGIC_FLOOR_EXIT),1) FPL26_ILS_HURDLE_CONTINUE=$(if $(ILS_HURDLE),$(ILS_HURDLE),1) FPL26_ILS_MEASURED_PRIORS=$(if $(MEASURED_PRIORS),$(MEASURED_PRIORS),1) FPL26_PHYSOPT_DEFAULT_FIXPOINT=$(if $(PODF),$(PODF),0) FPL26_ILS_LADDER_ORDER_BY_WNS=$(if $(LADDER_WNS),$(LADDER_WNS),1) FPL26_RECIPE_PASS=$(if $(RECIPE_PASS),$(RECIPE_PASS),1) FPL26_RECIPE_FIRST_DEEP=$(if $(RECIPE_FIRST_DEEP),$(RECIPE_FIRST_DEEP),1) FPL26_SUBBAND_PHYSOPT_FLOOR=$(if $(SUBBAND_PHYSOPT_FLOOR),$(SUBBAND_PHYSOPT_FLOOR),1) FPL26_ETO_RETIME_CANDIDATE=$(if $(ETO_RETIME),$(ETO_RETIME),1) FPL26_MIDBAND_RETRY_HOLD=$(if $(MIDBAND_RETRY_HOLD),$(MIDBAND_RETRY_HOLD),1) FPL26_MIDBAND_ROUTE_RUNG=$(if $(MIDBAND_ROUTE_RUNG),$(MIDBAND_ROUTE_RUNG),1) FPL26_OWNFRONT_RETIME_CANDIDATE=$(if $(OWNFRONT_RETIME),$(OWNFRONT_RETIME),1) FPL26_MUX_MD5_TRUST=$(if $(MUX_MD5_TRUST),$(MUX_MD5_TRUST),1) FPL26_SHALLOW_DETERMINIZER_CANDIDATE=$(if $(SHALLOW_DET),$(SHALLOW_DET),1) FPL26_POSITIVE_SLACK_CONTINUE=$(if $(POSITIVE_SLACK),$(POSITIVE_SLACK),1) $(PYTHON) dcp_optimizer.py "$(DCP)" --contest-mode --llm-cost-budget 0.10 --phase1-timeout-scale $(if $(PHASE1_SCALE),$(PHASE1_SCALE),3.0) $(if $(filter 0,$(ILS)),,--ils-polish) $(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback) $(if $(POLISH_RESERVE_S),--polish-reserve-s $(POLISH_RESERVE_S)) --max-wall-seconds $(if $(MAX_WALL),$(MAX_WALL),3500)

# SHIP FLAGS: the FPL26_* env vars injected below arm the shipped
# configuration (recipe-pass candidates, retime fronts, deep-replace,
# floor exits, MUX md5 trust, ...).  Each has an off-knob make var and a
# runtime FPL26_NO_* kill switch where noted.  The full reference with
# one-line descriptions and the Makefile-vs-code-default layering rules
# lives in docs/CONFIGURATION.md; scripts/ship_config.py prints the
# composed effective config.  The vars are injected on BOTH launch
# branches (wrapper and the `|| ` safety net) so the two can never
# silently diverge.
# NOTE: run_optimizer (the target the contest eval invokes) now runs the
# variance-protected multi-restart wrapper (best of N attempts within the wall
# budget, keep best valid DCP, cost-capped to the $1/benchmark budget).
# β circuit-breaker: COST_CEILING (default 0.80) is the hard CUMULATIVE
# LLM-spend ceiling across attempts — predictive pre-launch gate + shrinking
# per-attempt LLM_COST_BUDGET (the contest eval ZEROES a benchmark at $1.00
# cumulative spend). COST_CEILING=0 disables (kill switch).
# Last-resort attempt: when no attempt produced usable output, the
# wrapper now runs its OWN budget-aware last-resort contest-mode attempt
# (LLM_COST_BUDGET = max($0.01, ceiling − spent) — only the wrapper knows
# cumulative spend), so a valid DCP is still always emitted WITHOUT handing a
# fresh unbudgeted $0.75 allowance to a post-ceiling retry. The `||` above is
# a pure safety net that only fires when the wrapper CRASHED PRE-PYTHON
# (interpreter/import failure), where spend is unknowable — hence the small
# fixed --llm-cost-budget 0.10 belt-and-suspenders. For a quick single dev
# run use `make run_once`.
# C1-T3 post-route polish reserve: POLISH_RESERVE_S (agent default 500s;
# 0 disables) fences speculative routed-state-destroying dispatch out of the
# last N seconds once a routed banked best exists, so the final post-route
# phys_opt polish is affordable by construction (official beta boom_soc_v2:
# polish refused at est 600s > 443s remaining). Plumbed as env
# FPL26_POLISH_RESERVE_S through the wrapper (each attempt's agent reads it;
# CLI --polish-reserve-s wins over env). Orthogonal to COST_CEILING (wall
# seconds vs LLM $ — the two gates never couple).

# Run optimizer with contest-mode hygiene (hidden contest designs).
# Always passes --contest-mode.  Strategy-memory retrieval is
# fingerprint-only in every mode; contest mode additionally appends the
# negative-memory advisory block.  Default flow (run_optimizer) stays
# unchanged.
#
# Usage:
#   make run_once DCP=path/to/input.dcp \
#       [OUTPUT=path/to/output.dcp] [MAX_WALL=1800]
#
# PathGuard remains enforce by default.  decisions.jsonl is always
# emitted under dcp_optimizer_run-<ts>/.  OUTPUT defaults to /tmp so
# the submission tree is NEVER written by accident.
run_once:
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_once DCP=input.dcp [OUTPUT=output.dcp] [MAX_WALL=1800]"; \
		exit 1; \
	fi
	@if [ ! -f "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP file not found: $(DCP)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@_BASENAME=$$(basename "$(DCP)" .dcp); \
	_OUTPUT="$(OUTPUT)"; \
	if [ -z "$$_OUTPUT" ]; then _OUTPUT="/tmp/contestmode_$${_BASENAME}_optimized.dcp"; fi; \
	printf "$(COLOR_GREEN)Running CONTEST-MODE optimizer on $(DCP)...$(COLOR_RESET)\n"; \
	printf "$(COLOR_GREEN)  output: %s$(COLOR_RESET)\n" "$$_OUTPUT"; \
	if [ -n "$(MAX_WALL)" ]; then \
		printf "$(COLOR_GREEN)  max-wall-seconds: %s$(COLOR_RESET)\n" "$(MAX_WALL)"; \
	fi; \
	if ! command -v java >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)Java not found on PATH, attempting to use Java from Vivado...$(COLOR_RESET)\n"; \
		VIVADO_PATH=$$(command -v $(VIVADO_EXEC) 2>/dev/null); \
		if [ -n "$$VIVADO_PATH" ]; then \
			VIVADO_BIN_DIR=$$(dirname $$VIVADO_PATH); \
			VIVADO_ROOT=$$(dirname $$VIVADO_BIN_DIR); \
			JAVA_FOUND=$$(ls $$VIVADO_ROOT/tps/lnx64/jre11*/bin/java 2>/dev/null | head -n 1); \
			if [ -z "$$JAVA_FOUND" ]; then \
				JAVA_FOUND=$$(ls $$VIVADO_ROOT/tps/lnx64/jre*/bin/java 2>/dev/null | head -n 1); \
			fi; \
			if [ -n "$$JAVA_FOUND" ]; then \
				export JAVA_HOME=$$(dirname $$(dirname $$JAVA_FOUND)); \
				export PATH="$$JAVA_HOME/bin:$$PATH"; \
				printf "$(COLOR_GREEN)Using Java from Vivado: %s$(COLOR_RESET)\n" "$$JAVA_HOME"; \
			fi; \
		fi; \
	fi; \
	echo ""; \
	$(PYTHON) dcp_optimizer.py "$(DCP)" --contest-mode \
		--output "$$_OUTPUT" \
		--phase1-timeout-scale $(if $(PHASE1_SCALE),$(PHASE1_SCALE),3.0) \
		$(if $(filter 0,$(ILS)),,--ils-polish) \
		$(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback) \
		$(if $(LLM_COST_BUDGET),--llm-cost-budget $(LLM_COST_BUDGET)) \
		$(if $(POLISH_RESERVE_S),--polish-reserve-s $(POLISH_RESERVE_S)) \
		$(if $(MAX_WALL),--max-wall-seconds $(MAX_WALL))

# PHASE1_SCALE (default 3.0) multiplies Phase-1 step timeouts (open_checkpoint,
# report_timing_summary, high-fanout, spread). Large contest DCPs (ispd16 152MB,
# boom_soc) can exceed the base open_checkpoint timeout under disk-I/O load and
# fail Phase 1 -> instant 0 regardless of the optimization recipe (observed
# on ispd16). The scale is a CAP, not added latency: small designs
# finish Phase 1 fast so the higher cap never binds. Validated: ispd16 at
# scale 3.0 opens cleanly and reaches its retiming win (+16.87 MHz).

# Multi-restart, keep-best-valid wrapper (variance defense).
# Runs the UNCHANGED agent (via run_once) multiple times within
# the wall budget and keeps the best valid DCP. Defends against single-shot
# LLM-path variance (a design can draw 0 or its full gain across runs;
# best-of-N reliably captures the win). Cost-capped to respect the eval's
# $1/benchmark budget. Output -> eval-expected <input_dir>/<stem>_optimized.dcp.
#
# Usage:
#   make run_optimizer_multirestart DCP=path/to/input.dcp \
#       [OUTPUT=final.dcp] [MAX_WALL=3500] [MAX_ATTEMPTS=4] [COST_CAP=0.85]
#
# NOTE: recommended (variance-protected) eval entrypoint. Wiring
# `run_optimizer` -> this is the pre-submission step.
run_optimizer_multirestart:
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_optimizer_multirestart DCP=input.dcp [OUTPUT=final.dcp] [MAX_WALL=3500] [MAX_ATTEMPTS=4] [COST_CAP=0.85]"; \
		exit 1; \
	fi
	$(if $(POLISH_RESERVE_S),FPL26_POLISH_RESERVE_S=$(POLISH_RESERVE_S)) $(PYTHON) scripts/multi_restart_optimize.py "$(DCP)" \
		$(if $(OUTPUT),--final-output "$(OUTPUT)") \
		--total-wall $(if $(MAX_WALL),$(MAX_WALL),3500) \
		--max-attempts $(if $(MAX_ATTEMPTS),$(MAX_ATTEMPTS),4) \
		--cost-cap $(if $(COST_CAP),$(COST_CAP),0.85) \
		--cost-ceiling $(if $(COST_CEILING),$(COST_CEILING),0.80)

# Run test mode: Run dcp_optimizer.py with --test flag (no LLM required)
run_no_llm:
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_no_llm DCP=input.dcp"; \
		echo ""; \
		echo "Supported example DCPs:"; \
		echo "  make run_no_llm DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp      # Pblock optimization"; \
		echo "  make run_no_llm DCP=fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp   # Cell re-placement"; \
		exit 1; \
	fi
	@if [ ! -f "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP file not found: $(DCP)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_GREEN)Running optimizer in TEST MODE on $(DCP)...$(COLOR_RESET)\n"
	@# Set up Java from Vivado if Java is not available
	@if ! command -v java >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)Java not found on PATH, attempting to use Java from Vivado...$(COLOR_RESET)\n"; \
		VIVADO_PATH=$$(command -v $(VIVADO_EXEC) 2>/dev/null); \
		if [ -n "$$VIVADO_PATH" ]; then \
			VIVADO_BIN_DIR=$$(dirname $$VIVADO_PATH); \
			VIVADO_ROOT=$$(dirname $$VIVADO_BIN_DIR); \
			VIVADO_JAVA="$$VIVADO_ROOT/tps/lnx64/jre11*/bin/java"; \
			if ls $$VIVADO_JAVA >/dev/null 2>&1; then \
				JAVA_FOUND=$$(ls $$VIVADO_JAVA | head -n 1); \
				export JAVA_HOME=$$(dirname $$(dirname $$JAVA_FOUND)); \
				export PATH="$$JAVA_HOME/bin:$$PATH"; \
				printf "$(COLOR_GREEN)Using Java from Vivado: %s$(COLOR_RESET)\n" "$$JAVA_HOME"; \
			fi; \
		fi; \
	fi; \
	echo ""; \
	$(PYTHON) dcp_optimizer.py "$(DCP)" --test $(if $(MAX_NETS),--max-nets $(MAX_NETS))

# Validation target: Validate functional equivalence between two DCPs
validate:
	@printf "$(COLOR_BLUE)╔══════════════════════════════════════════════════════════════════╗$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)║         DCP Equivalence Validation (2-Phase Approach)            ║$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)╚══════════════════════════════════════════════════════════════════╝$(COLOR_RESET)\n"
	@echo ""
	@# Check if GOLDEN and REVISED are provided
	@if [ -z "$(GOLDEN)" ]; then \
		printf "$(COLOR_RED)✗ Error: GOLDEN DCP not specified$(COLOR_RESET)\n"; \
		echo "Usage: make validate GOLDEN=<golden.dcp> REVISED=<revised.dcp> [VECTORS=200]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make validate GOLDEN=logicnets_jscl.dcp REVISED=logicnets_jscl_optimized.dcp"; \
		exit 1; \
	fi
	@if [ -z "$(REVISED)" ]; then \
		printf "$(COLOR_RED)✗ Error: REVISED DCP not specified$(COLOR_RESET)\n"; \
		echo "Usage: make validate GOLDEN=<golden.dcp> REVISED=<revised.dcp> [VECTORS=200]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make validate GOLDEN=logicnets_jscl.dcp REVISED=logicnets_jscl_optimized.dcp"; \
		exit 1; \
	fi
	@# Check if files exist
	@if [ ! -f "$(GOLDEN)" ]; then \
		printf "$(COLOR_RED)✗ Error: Golden DCP not found: $(GOLDEN)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@if [ ! -f "$(REVISED)" ]; then \
		printf "$(COLOR_RED)✗ Error: Revised DCP not found: $(REVISED)$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@# Run validation
	@printf "$(COLOR_GREEN)Golden DCP:$(COLOR_RESET)  $(GOLDEN)\n"
	@printf "$(COLOR_GREEN)Revised DCP:$(COLOR_RESET) $(REVISED)\n"
	@printf "$(COLOR_GREEN)Test Vectors:$(COLOR_RESET) $(or $(VECTORS),200)\n"
	@echo ""
	@if [ -n "$(VECTORS)" ]; then \
		$(PYTHON) validate_dcps.py "$(GOLDEN)" "$(REVISED)" --vectors $(VECTORS); \
	else \
		$(PYTHON) validate_dcps.py "$(GOLDEN)" "$(REVISED)"; \
	fi

# Quick validation example using demo DCPs
validate_demo:
	@printf "$(COLOR_BLUE)╔══════════════════════════════════════════════════════════════════╗$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)║                  Validation Demo (Simulated)                     ║$(COLOR_RESET)\n"
	@printf "$(COLOR_BLUE)╚══════════════════════════════════════════════════════════════════╝$(COLOR_RESET)\n"
	@echo ""
	@echo "This demo validates a DCP against itself (should always PASS)."
	@echo "For real validation, first optimize a design, then validate:"
	@echo ""
	@echo "  1. python dcp_optimizer.py design.dcp --output design_optimized.dcp"
	@echo "  2. make validate GOLDEN=design.dcp REVISED=design_optimized.dcp"
	@echo ""
	@# Check if example DCP exists
	@if [ ! -f "$(EXAMPLE_DCP_2)" ]; then \
		printf "$(COLOR_YELLOW)Example DCP not found, downloading...$(COLOR_RESET)\n"; \
		$(MAKE) download_dcps; \
	fi
	@# For demo, validate DCP against itself (should always pass)
	@printf "$(COLOR_GREEN)Running demo validation (self-check)...$(COLOR_RESET)\n"
	@echo ""
	$(PYTHON) validate_dcps.py "$(EXAMPLE_DCP_2)" "$(EXAMPLE_DCP_2)" --vectors 1000

# Build a strict, leak-proof contest submission archive (.tar.gz, which the
# harness accepts alongside .zip). Excludes .env/.venv/.git/run-dirs/benchmarks
# and VERIFIES no secrets leaked + that the archive imports cleanly (catches a
# missing untracked module before the contest harness does). Output defaults to
# /tmp/fpl26_submission.tar.gz; override with OUT=path.
#   make submission                 # -> /tmp/fpl26_submission.tar.gz
#   make submission OUT=./final.tar.gz
submission:
	@bash scripts/build_submission.sh $(if $(OUT),$(OUT),/tmp/fpl26_submission.tar.gz)

# Submission validator (CI gate) — runs the full per-design contest-clock
# validation pipeline. Hard rule: no validator pass, no claimed MHz. Fails
# non-zero if any DCP regressed or failed.
#
# Needs a `submission/` directory that this repository does not ship: a
# MANIFEST.tsv listing the designs and a package_and_validate.sh to drive
# them. It is kept because it is the gate the scored results were held to;
# assemble that directory to use it, or use `make validate` for a single
# golden/revised pair.
# Run the offline test suite (no Vivado, no LLM, no network needed).
# tests/ is the main suite; module-local tests in optimizer/, scheduler/
# and recipes/ are included.  VivadoMCP/ and RapidWrightMCP/ tests need
# the live tools and are NOT part of this offline invocation.
# Offline suite: no Vivado, no network, no API key.
#
# The suite writes run-dir and strategy-memory residue into the repo root.
# Only what THIS invocation created is removed afterwards — a blanket
# `rm -rf dcp_optimizer_run-*` would delete a real run's logs and the
# accumulated strategy memory of anyone who ran the optimizer here first.
test:
	@_before=$$(ls -d dcp_optimizer_run-* 2>/dev/null | sort); \
	_mem_existed=$$([ -f strategy_memory.jsonl ] && echo yes || echo no); \
	$(PYTHON) -m pytest tests/ optimizer/ scheduler/ recipes/ tests/test_validate_dcps.py -q; \
	rc=$$?; \
	for d in $$(ls -d dcp_optimizer_run-* 2>/dev/null | sort); do \
		echo "$$_before" | grep -qxF "$$d" || rm -rf "$$d"; \
	done; \
	[ "$$_mem_existed" = yes ] || rm -f strategy_memory.jsonl; \
	exit $$rc

# Narrow lint gate (see ruff.toml for what is selected and why, and for the
# known backlog that is not). Green as of the commit that added it, so a red
# result means something regressed. Not part of `make test`: the offline suite
# must stay runnable with the three packages in requirements-dev.txt.
lint:
	@command -v ruff >/dev/null 2>&1 || { \
		printf "$(COLOR_YELLOW)ruff not found. pip install -r requirements-lint.txt$(COLOR_RESET)\n"; \
		exit 1; \
	}
	ruff check .

# Clean target: Remove run directories and Vivado-generated .Xil directories
clean:
	@printf "$(COLOR_YELLOW)Cleaning generated files...$(COLOR_RESET)\n"
	@# Remove run directories (contain all logs, journals, intermediate files)
	@if ls dcp_optimizer_run-* >/dev/null 2>&1; then \
		rm -rf dcp_optimizer_run-*; \
		echo "Removed dcp_optimizer_run-* directories"; \
	fi
	@# Remove .Xil directories (Vivado generates these outside run directories)
	@if [ -d ".Xil" ]; then \
		rm -rf .Xil; \
		echo "Removed .Xil/"; \
	fi
	@if [ -d "VivadoMCP/.Xil" ]; then \
		rm -rf VivadoMCP/.Xil; \
		echo "Removed VivadoMCP/.Xil/"; \
	fi
	@printf "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)\n"
	@echo "Note: Optimized DCP files were preserved"

# Very clean target: Clean + remove __pycache__ and example DCPs
veryclean: clean
	@printf "$(COLOR_YELLOW)Performing deep clean...$(COLOR_RESET)\n"
	@# Remove Python cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Removed __pycache__ directories"
	@# Remove benchmark directory and tarball
	@rm -rf $(BENCHMARK_DIR) $(BENCHMARK_TARBALL)
	@echo "Removed benchmark directory and tarball"
	@printf "$(COLOR_GREEN)✓ Deep clean complete$(COLOR_RESET)\n"
