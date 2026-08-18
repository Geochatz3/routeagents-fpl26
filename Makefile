# Makefile for FPGA Design Optimization Agent

# Configuration
PYTHON := python3
PIP := $(PYTHON) -m pip

# Vivado executable - can be overridden with: make setup VIVADO_EXEC=/path/to/vivado
VIVADO_EXEC ?= vivado
export VIVADO_EXEC

# Set JAVA_HOME from PATH or Vivado if not already set
# Python RapidWright may need JAVA_HOME to be set, but often users only have `java` on PATH
# See: https://www.rapidwright.io/docs/Install.html#using-java-distributed-with-vivado
ifndef JAVA_HOME
  # ORDER MATTERS (aug08): prefer Vivado's bundled JRE over a PATH java.
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

.PHONY: setup build-rapidwright run_optimizer run_test validate validate_demo validate-submission run-submission submission clean veryclean help

# Default target
help:
	@echo "FPGA Design Optimization Agent - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup              - Install dependencies, build RapidWright, download example DCPs"
	@echo "  build-rapidwright  - Build RapidWright from source (git submodule)"
	@echo "  run_optimizer      - Run optimizer on a DCP file (LLM-guided, requires API key)"
	@echo "  run_test           - Run optimizer in test mode (no LLM, hardcoded optimization)"
	@echo "  validate           - Validate functional equivalence between two DCPs"
	@echo "  validate_demo      - Run validation demo (self-check)"
	@echo "  clean              - Remove generated files (run directories, logs, Vivado outputs)"
	@echo "  veryclean          - Remove all generated files including example DCPs"
	@echo ""
	@echo "Usage examples:"
	@echo "  make setup"
	@echo "  make setup VIVADO_EXEC=/tools/Xilinx/Vivado/2025.2/bin/vivado"
	@echo "  make run_optimizer DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp"
	@echo "  make run_test DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp"
	@echo "  make run_test DCP=fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp"
	@echo "  make validate GOLDEN=design.dcp REVISED=design_optimized.dcp"
	@echo "  make validate GOLDEN=design.dcp REVISED=design_optimized.dcp VECTORS=50000"
	@echo "  make validate_demo"
	@echo "  make clean"
	@echo ""
	@echo "Environment variables:"
	@echo "  VIVADO_EXEC     - Path to Vivado executable (default: vivado)"
	@echo "  JAVA_HOME       - Java installation directory (auto-detected from PATH if not set)"
	@echo "  DCP             - Input DCP file for run_optimizer / run_test targets"
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
	
	@printf "$(COLOR_GREEN)===== Setup Complete! =====$(COLOR_RESET)\n"
	@echo ""
	@echo "Next steps - run the optimizer:"
	@echo ""
	@echo "  Test mode (no API key required):"
	@echo "    make run_test DCP=$(EXAMPLE_DCP_1)"
	@echo "    make run_test DCP=$(EXAMPLE_DCP_2)"
	@echo ""
	@echo "  Full LLM-guided optimizer (requires OPENROUTER_API_KEY):"
	@echo "    make run_optimizer DCP=$(EXAMPLE_DCP_1)"
	@echo ""
	@echo "Output will be in:"
	@echo "  - Optimized DCP: <input_name>_optimized-<timestamp>.dcp"
	@echo "  - Run logs: dcp_optimizer_run-<timestamp>/"
	@echo ""

# Build RapidWright from source (git submodule).
#
# When this repo is unpacked from a release archive (no .git), the source
# build is unavailable AND unnecessary: the pip-installed `rapidwright`
# package ships its own bundled jars and is what RapidWrightMCP actually
# imports at runtime.  Skip the source build in that case.
#
# Recipe is a single shell so early-exit propagates correctly.
#
# Skip conditions (any one is enough):
#   1. `.git` does not exist (flat archive, e.g. submission tarball)
#   2. submodule init fails for any reason (no network, etc.)
#
# When skipped, RAPIDWRIGHT_PATH/CLASSPATH from the environment are still
# exported (harmless empty globs); pip rapidwright uses its own paths.
build-rapidwright:
	@set -e; \
	printf "$(COLOR_YELLOW)Building RapidWright from source...$(COLOR_RESET)\n"; \
	if [ ! -f "$(RAPIDWRIGHT_PATH)/gradlew" ]; then \
		if [ ! -d ".git" ]; then \
			printf "$(COLOR_YELLOW)⚠ No .git and no $(RAPIDWRIGHT_PATH)/gradlew — skipping source build.$(COLOR_RESET)\n"; \
			printf "$(COLOR_YELLOW)  pip-installed rapidwright will be used at runtime.$(COLOR_RESET)\n"; \
			exit 0; \
		fi; \
		printf "$(COLOR_YELLOW)Initializing RapidWright git submodule...$(COLOR_RESET)\n"; \
		if ! git submodule update --init RapidWright; then \
			printf "$(COLOR_YELLOW)⚠ Submodule init failed — skipping source build (pip rapidwright will be used).$(COLOR_RESET)\n"; \
			exit 0; \
		fi; \
		if [ ! -f "$(RAPIDWRIGHT_PATH)/gradlew" ]; then \
			printf "$(COLOR_YELLOW)⚠ gradlew still missing after submodule init — skipping (pip rapidwright will be used).$(COLOR_RESET)\n"; \
			exit 0; \
		fi; \
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
	$(if $(POLISH_RESERVE_S),FPL26_POLISH_RESERVE_S=$(POLISH_RESERVE_S)) FPL26_DEEP_WNS_TAIL_RESERVE=$(if $(DEEP_WNS_TAIL_RESERVE),$(DEEP_WNS_TAIL_RESERVE),2400) FPL26_DEEP_REPLACE=$(if $(DEEP_REPLACE),$(DEEP_REPLACE),1) FPL26_DEEP_REPLACE_FIRST=$(if $(DEEP_REPLACE_FIRST),$(DEEP_REPLACE_FIRST),1) FPL26_DEEP_REPLACE_UNBANDED=$(if $(DEEP_REPLACE_UNBANDED),$(DEEP_REPLACE_UNBANDED),1) FPL26_DEEP_FIRST_SIZEGATED=$(if $(DEEP_FIRST_SIZEGATED),$(DEEP_FIRST_SIZEGATED),1) FPL26_DEEP_REPLACE_B3=$(if $(DEEP_REPLACE_B3),$(DEEP_REPLACE_B3),1) FPL26_B3_FLOOR_EXIT=$(if $(B3_FLOOR_EXIT),$(B3_FLOOR_EXIT),1) FPL26_LOGIC_FLOOR_EXIT=$(if $(LOGIC_FLOOR_EXIT),$(LOGIC_FLOOR_EXIT),1) FPL26_ILS_HURDLE_CONTINUE=$(if $(ILS_HURDLE),$(ILS_HURDLE),1) FPL26_ILS_MEASURED_PRIORS=$(if $(MEASURED_PRIORS),$(MEASURED_PRIORS),1) FPL26_PHYSOPT_DEFAULT_FIXPOINT=$(if $(PODF),$(PODF),0) FPL26_ILS_LADDER_ORDER_BY_WNS=$(if $(LADDER_WNS),$(LADDER_WNS),1) FPL26_RECIPE_PASS=$(if $(RECIPE_PASS),$(RECIPE_PASS),1) FPL26_RECIPE_FIRST_DEEP=$(if $(RECIPE_FIRST_DEEP),$(RECIPE_FIRST_DEEP),1) FPL26_FIR_SUBBAND_FLOOR=$(if $(FIR_SUBBAND_FLOOR),$(FIR_SUBBAND_FLOOR),1) FPL26_VEX2_RETIME_CANDIDATE=$(if $(VEX2_RETIME),$(VEX2_RETIME),1) FPL26_MINIISP_RETRY_HOLD=$(if $(MINIISP_RETRY_HOLD),$(MINIISP_RETRY_HOLD),1) FPL26_CORESCORE_ROUTE_RUNG=$(if $(CORESCORE_ROUTE_RUNG),$(CORESCORE_ROUTE_RUNG),1) FPL26_OWNFRONT_RETIME_CANDIDATE=$(if $(OWNFRONT_RETIME),$(OWNFRONT_RETIME),1) FPL26_MUX_MD5_TRUST=$(if $(MUX_MD5_TRUST),$(MUX_MD5_TRUST),1) FPL26_SPAM_DETERMINIZER_CANDIDATE=$(if $(SPAM_DET),$(SPAM_DET),1) FPL26_POSITIVE_SLACK_CONTINUE=$(if $(POSITIVE_SLACK),$(POSITIVE_SLACK),1) $(PYTHON) scripts/multi_restart_optimize.py "$(DCP)" \
		--total-wall $(if $(MAX_WALL),$(MAX_WALL),3500) \
		--max-attempts $(if $(MAX_ATTEMPTS),$(MAX_ATTEMPTS),4) \
		--cost-cap $(if $(COST_CAP),$(COST_CAP),0.85) \
		--cost-ceiling $(if $(COST_CEILING),$(COST_CEILING),0.80) \
		$(if $(filter 0,$(ILS)),,--ils-polish) \
		$(if $(filter 1 true yes on,$(SPLIT_AWARE)),--split-aware) \
		$(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback) \
		|| FPL26_DEEP_WNS_TAIL_RESERVE=$(if $(DEEP_WNS_TAIL_RESERVE),$(DEEP_WNS_TAIL_RESERVE),2400) FPL26_DEEP_REPLACE=$(if $(DEEP_REPLACE),$(DEEP_REPLACE),1) FPL26_DEEP_REPLACE_FIRST=$(if $(DEEP_REPLACE_FIRST),$(DEEP_REPLACE_FIRST),1) FPL26_DEEP_REPLACE_UNBANDED=$(if $(DEEP_REPLACE_UNBANDED),$(DEEP_REPLACE_UNBANDED),1) FPL26_DEEP_FIRST_SIZEGATED=$(if $(DEEP_FIRST_SIZEGATED),$(DEEP_FIRST_SIZEGATED),1) FPL26_DEEP_REPLACE_B3=$(if $(DEEP_REPLACE_B3),$(DEEP_REPLACE_B3),1) FPL26_B3_FLOOR_EXIT=$(if $(B3_FLOOR_EXIT),$(B3_FLOOR_EXIT),1) FPL26_LOGIC_FLOOR_EXIT=$(if $(LOGIC_FLOOR_EXIT),$(LOGIC_FLOOR_EXIT),1) FPL26_ILS_HURDLE_CONTINUE=$(if $(ILS_HURDLE),$(ILS_HURDLE),1) FPL26_ILS_MEASURED_PRIORS=$(if $(MEASURED_PRIORS),$(MEASURED_PRIORS),1) FPL26_PHYSOPT_DEFAULT_FIXPOINT=$(if $(PODF),$(PODF),0) FPL26_ILS_LADDER_ORDER_BY_WNS=$(if $(LADDER_WNS),$(LADDER_WNS),1) FPL26_RECIPE_PASS=$(if $(RECIPE_PASS),$(RECIPE_PASS),1) FPL26_RECIPE_FIRST_DEEP=$(if $(RECIPE_FIRST_DEEP),$(RECIPE_FIRST_DEEP),1) FPL26_FIR_SUBBAND_FLOOR=$(if $(FIR_SUBBAND_FLOOR),$(FIR_SUBBAND_FLOOR),1) FPL26_VEX2_RETIME_CANDIDATE=$(if $(VEX2_RETIME),$(VEX2_RETIME),1) FPL26_MINIISP_RETRY_HOLD=$(if $(MINIISP_RETRY_HOLD),$(MINIISP_RETRY_HOLD),1) FPL26_CORESCORE_ROUTE_RUNG=$(if $(CORESCORE_ROUTE_RUNG),$(CORESCORE_ROUTE_RUNG),1) FPL26_OWNFRONT_RETIME_CANDIDATE=$(if $(OWNFRONT_RETIME),$(OWNFRONT_RETIME),1) FPL26_MUX_MD5_TRUST=$(if $(MUX_MD5_TRUST),$(MUX_MD5_TRUST),1) FPL26_SPAM_DETERMINIZER_CANDIDATE=$(if $(SPAM_DET),$(SPAM_DET),1) FPL26_POSITIVE_SLACK_CONTINUE=$(if $(POSITIVE_SLACK),$(POSITIVE_SLACK),1) $(PYTHON) dcp_optimizer.py "$(DCP)" --contest-mode --llm-cost-budget 0.10 --phase1-timeout-scale $(if $(PHASE1_SCALE),$(PHASE1_SCALE),3.0) $(if $(filter 0,$(ILS)),,--ils-polish) $(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback) $(if $(POLISH_RESERVE_S),--polish-reserve-s $(POLISH_RESERVE_S)) --max-wall-seconds $(if $(MAX_WALL),$(MAX_WALL),3500)

# v4.0 LEVERS (v40-levers, aug05 — PREREG_V40_LEVERS_aug05.md): three
# DEFAULT-OFF python flags armed here on BOTH launch branches (jul30
# "Makefile not in the ship surface" lesson):
#   FPL26_VEX2_RETIME_CANDIDATE  (off-knob VEX2_RETIME=0) — q07 retime
#     chain as a SECOND shallow-pass MUX candidate, |wns_in| [0.60,1.05].
#   FPL26_MINIISP_RETRY_HOLD     (off-knob MINIISP_RETRY_HOLD=0) —
#     mid-band-scoped retry-baseline-gate (global env flag untouched).
#   FPL26_CORESCORE_ROUTE_RUNG   (off-knob CORESCORE_ROUTE_RUNG=0) —
#     mid-band post-loop route-Explore MUX candidate.
# v4.1 REV2 (v41-eggs, aug05 — PREREG_V41_REV2_aug05.md), same contract:
#   FPL26_OWNFRONT_RETIME_CANDIDATE (off-knob OWNFRONT_RETIME=0) — the
#     UNIFIED own-front retime candidate (supersedes the v4.1-eggs
#     WLD_RETIME stacking design): ONE second shallow candidate, front
#     selected per run (|wns_in| [0.60,1.00) -> ETO/q07 chain,
#     [1.00,1.05] -> WLD chain); when armed the vex2 candidate defers
#     (reason=ownfront_supersedes) so ONE retime candidate spends wall
#     per run.  Off/killed -> exact v4.0.1 behavior (vex2 runs).
#   FPL26_MUX_MD5_TRUST (off-knob MUX_MD5_TRUST=0) — finalize MUX trusts
#     the registration-time re-measure when the winning candidate's
#     md5+size still match (skips the 120s-budget re-open that twice ate
#     a verified +92.38-class digit winner); mismatch falls back to the
#     full structural validate unchanged.
# v4.1.2 (v41-eggs, aug06 — PREREG_V41_REV2_aug05.md v4.1.2 section):
#   FPL26_SPAM_DETERMINIZER_CANDIDATE (off-knob SPAM_DET=0; runtime kill
#     switch FPL26_NO_SPAM_DETERMINIZER_CANDIDATE) — the box2-drilled
#     spam determinizer chain (place ASL_medium -> route Explore ->
#     route AE incremental -> phys_opt AFWR; -0.543/466.64 x3
#     bit-identical) as a THIRD shallow-pass MUX candidate on
#     |wns_in| [0.60, 0.90) — the exact window the ownfront v4.1.1
#     floor vacated.  SWAP (a5dcddb): in [0.60,0.90) the vex2
#     candidate defers to this one (measured MUX-discarded x3 there);
#     floors the spam-class timing lottery at the 29.19 class when its
#     own wall gate (2225 s) funds it.
# STAGING ONLY until the prereg'd A/B + full-16 parity pass; the
# ship decision is made there, not here.
# FPL26_RECIPE_FIRST_DEEP (staging/v30-rfd, aug04 — BLOCKER-2): arms the
# size-anchored PRE-LLM deep-band recipe gate on BOTH launch branches
# (wrapper + `||` safety net — the jul30 "Makefile not in the ship
# surface" lesson: one branch armed alone measures nothing).  Requires
# FPL26_RECIPE_PASS (armed above, 2a7b207); the runtime kill switch
# FPL26_NO_RECIPE_PASS kills both.  Off-knob: RECIPE_FIRST_DEEP=0.
# Cited basis: adv_review_qwen_deep_aug04.md conditional-GO (dynamic
# abort + anchor floor + scaled LLM floor SHIPPED, cd2eaec lineage) +
# the aug04 panel/prereg gate — STAGING ONLY until tonight's sweep
# readout + parity fire pass; the ship decision is made there, not here.
# NOTE: run_optimizer (the target the contest eval invokes) now runs the
# variance-protected multi-restart wrapper (best of N attempts within the wall
# budget, keep best valid DCP, cost-capped to the $1/benchmark budget).
# C1-T1 β circuit-breaker: COST_CEILING (default 0.80) is the hard CUMULATIVE
# LLM-spend ceiling across attempts — predictive pre-launch gate + shrinking
# per-attempt LLM_COST_BUDGET (the eval ZEROES a benchmark at $1.00 cumulative;
# preview #15 / reh-1 fir $0.76). COST_CEILING=0 disables (kill switch).
# FIX 2 (S2, jul20 C1 review): when no attempt produced usable output, the
# wrapper now runs its OWN budget-aware last-resort contest-mode attempt
# (LLM_COST_BUDGET = max($0.01, ceiling − spent) — only the wrapper knows
# cumulative spend), so a valid DCP is still always emitted WITHOUT handing a
# fresh unbudgeted $0.75 allowance to a post-ceiling retry. The `||` above is
# a pure safety net that only fires when the wrapper CRASHED PRE-PYTHON
# (interpreter/import failure), where spend is unknowable — hence the small
# fixed --llm-cost-budget 0.10 belt-and-suspenders. For a quick single dev
# run use `make run_optimizer_contest`.
# C1-T3 post-route polish reserve: POLISH_RESERVE_S (agent default 500s;
# 0 disables) fences speculative routed-state-destroying dispatch out of the
# last N seconds once a routed banked best exists, so the final post-route
# phys_opt polish is affordable by construction (official beta boom_soc_v2:
# polish refused at est 600s > 443s remaining). Plumbed as env
# FPL26_POLISH_RESERVE_S through the wrapper (each attempt's agent reads it;
# CLI --polish-reserve-s wins over env). Orthogonal to COST_CEILING (wall
# seconds vs LLM $ — the two gates never couple).

# Run optimizer with contest-mode hygiene (hidden contest designs).
# Always passes --contest-mode so DESIGN_NOTES + exact-name retrieval
# + benchmark-name recipe gates are disabled.  Default flow
# (run_optimizer) stays unchanged.
#
# Usage:
#   make run_optimizer_contest DCP=path/to/input.dcp \
#       [OUTPUT=path/to/output.dcp] [MAX_WALL=1800]
#
# PathGuard remains enforce by default.  decisions.jsonl is always
# emitted under dcp_optimizer_run-<ts>/.  OUTPUT defaults to /tmp so
# the submission tree is NEVER written by accident.
run_optimizer_contest:
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_optimizer_contest DCP=input.dcp [OUTPUT=output.dcp] [MAX_WALL=1800]"; \
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
# 2026-06-02 on ispd16). The scale is a CAP, not added latency: small designs
# finish Phase 1 fast so the higher cap never binds. Validated: ispd16 at
# scale 3.0 opens cleanly and reaches its retiming win (+16.87 MHz, R1).

# Multi-restart, keep-best-valid wrapper (variance defense).
# Runs the UNCHANGED agent (via run_optimizer_contest) multiple times within
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
# `run_optimizer` -> this is the pre-submission step (.planning/CONTEST_COMPLIANCE.md).
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
run_test:
	@if [ -z "$(DCP)" ]; then \
		printf "$(COLOR_RED)Error: DCP variable not set$(COLOR_RESET)\n"; \
		echo "Usage: make run_test DCP=input.dcp"; \
		echo ""; \
		echo "Supported example DCPs:"; \
		echo "  make run_test DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp      # Pblock optimization"; \
		echo "  make run_test DCP=fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp   # Cell re-placement"; \
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

run-submission:
	@echo "Running submission...[Will be implemented later]"

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
# validation pipeline. Post-2026-05-19 hard rule: no validator pass, no
# claimed MHz. Fails non-zero if any DCP regressed/failed.
validate-submission:
	@printf "$(COLOR_BLUE)══════ Submission Validator (CI Gate) ══════$(COLOR_RESET)\n"
	@if [ ! -f submission/MANIFEST.tsv ]; then \
		printf "$(COLOR_RED)submission/MANIFEST.tsv missing$(COLOR_RESET)\n"; exit 2; fi
	@bash submission/package_and_validate.sh
	@if [ ! -f submission/results.tsv ]; then \
		printf "$(COLOR_RED)submission/results.tsv not produced$(COLOR_RESET)\n"; exit 2; fi
	@bad=$$(awk -F'\t' 'NR>1 && $$8 !~ /^PASS$$|^BASELINE_FIX$$/ {print $$1 ": " $$8}' submission/results.tsv); \
	if [ -n "$$bad" ]; then \
		printf "$(COLOR_RED)FAIL — non-PASS verdicts present:$(COLOR_RESET)\n"; \
		echo "$$bad" | sed 's/^/  /'; exit 1; fi
	@total=$$(awk -F'\t' 'NR>1 {d=$$7+0; if (d>=0) s+=d} END {printf "%.2f", s}' submission/results.tsv); \
	printf "$(COLOR_GREEN)PASS — honest validated total: +$$total MHz$(COLOR_RESET)\n"

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
