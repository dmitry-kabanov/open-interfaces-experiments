exp ?= .

DATA_CSV := \
    $(exp)/_output/runtime_vs_resolution_python.csv \
    $(exp)/_output/runtime_vs_resolution_julia.csv \
    $(exp)/_output/runtime_vs_resolution_python_jl_diffeq_numba.csv

.PHONY: all
all: $(exp)/run.py $(DATA_CSV)
	python $<

$(exp)/_output/runtime_vs_resolution_python.csv: $(exp)/call_ivp_python.py
	python $<

$(exp)/_output/runtime_vs_resolution_julia.csv: $(exp)/call_ivp_julia.jl
	julia $<

$(exp)/_output/runtime_vs_resolution_python_jl_diffeq_numba.csv: $(exp)/call_jl_diffeq_from_python.py
	python $<

.PHONY: clean
clean:
	rm $(exp)/*.csv
