exp ?= .

DATA_TXT := $(exp)/_output/rhs_evals.txt
DATA_CSV := \
    $(exp)/_output/runtime_vs_resolution_python.csv \
    $(exp)/_output/runtime_vs_resolution_julia.csv \
    $(exp)/_output/runtime_vs_resolution_python_jl_diffeq.csv

.PHONY : all
all: rhs-evals runtimes

.PHONY : rhs-evals
rhs-evals : $(DATA_TXT)
	@echo ""
	cat $<

$(DATA_TXT) : $(exp)/run_rhs_eval.py
	python $<

$(exp)/run_rhs_eval.py: $(exp)/call_rhs_eval_python.py $(exp)/call_rhs_eval_julia.jl

.PHONY : runtimes
runtimes : $(exp)/run.py $(DATA_CSV)
	@echo ""
	python $<

$(exp)/_output/runtime_vs_resolution_python.csv : $(exp)/call_ivp_python.py
	python $<

$(exp)/_output/runtime_vs_resolution_julia.csv : $(exp)/call_ivp_julia.jl
	julia $<

$(exp)/_output/runtime_vs_resolution_python_jl_diffeq.csv : $(exp)/call_ivp_python_jl_diffeq.py
	python $<

.PHONY : clean
clean :
	rm $(exp)/*.csv
