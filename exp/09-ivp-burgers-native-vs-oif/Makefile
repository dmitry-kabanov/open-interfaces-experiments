exp ?= .

DATA_TXT := $(exp)/_output/rhs_evals.txt
DATA_CSV := \
    $(exp)/_output/runtime_vs_resolution_python.csv \
    $(exp)/_output/runtime_vs_resolution_julia.csv \
    $(exp)/_output/runtime_vs_resolution_python_jl_diffeq.csv \
    $(exp)/_output/runtime_vs_resolution_python_native.csv

.PHONY : all
all: rhs-evals runtimes
	@echo

.PHONY : rhs-evals
rhs-evals : $(DATA_TXT)
	@echo ""
	@echo "---"
	cat $<

$(DATA_TXT) : $(exp)/run_rhs_eval.py $(exp)/call_rhs_eval_python.py $(exp)/call_rhs_eval_julia.jl
	python $<

.PHONY : runtimes
runtimes : $(exp)/run.py $(DATA_CSV)
	@echo ""
	@echo "---"
	python $<

$(exp)/_output/runtime_vs_resolution_python.csv : $(exp)/call_ivp_python.py
	python $<

$(exp)/_output/runtime_vs_resolution_julia.csv : $(exp)/call_ivp_julia.jl
	julia $<

$(exp)/_output/runtime_vs_resolution_python_jl_diffeq.csv : $(exp)/call_ivp_python_jl_diffeq.py
	python $< jl_diffeq

$(exp)/_output/runtime_vs_resolution_python_native.csv : $(exp)/call_ivp_python_jl_diffeq.py
	python $< native

$(exp)/call_rhs_eval_julia.jl: $(exp)/rhsversions.jl ../../helpers.jl

$(exp)/call_rhs_eval_python.py: $(exp)/rhsversions.py ../../helpers.py

.PHONY : clean
clean :
	rm $(exp)/*.csv
