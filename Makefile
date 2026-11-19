PYTHON ?= python
OUTPUT ?= results
WORKERS ?= 4
RUN = $(PYTHON) -m src.simulation
ARGS = --output "$(OUTPUT)"

ifeq ($(OS),Windows_NT)
VENV_PYTHON = .venv/Scripts/python.exe
else
VENV_PYTHON = .venv/bin/python
endif

.PHONY: all simulate analyze figures clean test profile complexity benchmark benchmark-serial benchmark-parallel parallel performance stability-check reproduce

all:
	$(RUN) all $(ARGS)

simulate:
	$(RUN) simulate $(ARGS)

analyze:
	$(RUN) analyze $(ARGS)

figures:
	$(RUN) figures $(ARGS)

clean:
	$(RUN) clean $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -v

profile:
	$(PYTHON) -m scripts.baseline profile

complexity:
	$(PYTHON) -m scripts.baseline complexity

benchmark:
	$(PYTHON) -m scripts.benchmark
	$(PYTHON) -m scripts.benchmark_parallel

benchmark-serial:
	$(PYTHON) -m scripts.benchmark

parallel:
	$(RUN) all $(ARGS) --workers $(WORKERS)

benchmark-parallel:
	$(PYTHON) -m scripts.benchmark_parallel

performance:
	$(PYTHON) -m scripts.performance $(ARGS)

stability-check:
	$(PYTHON) -m scripts.stability_check $(ARGS) $(if $(LOG),--log "$(LOG)")

reproduce: $(VENV_PYTHON)
	"$(VENV_PYTHON)" -m pip install -r requirements.txt
	"$(VENV_PYTHON)" -m src.simulation all $(ARGS) --workers $(WORKERS)
	"$(VENV_PYTHON)" -m unittest discover -s tests -v
	"$(VENV_PYTHON)" -m scripts.stability_check $(ARGS) $(if $(LOG),--log "$(LOG)")

$(VENV_PYTHON):
	$(PYTHON) -m venv .venv
