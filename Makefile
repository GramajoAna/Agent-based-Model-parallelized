APP=mosquitas 
#APP=bichos mosquitas

THRUSTPATH=.
#THRUSTPATH=/share/apps/icnpg/common/thrust-master

RANDOM123=-I./include/

all: ${APP}

%: %.cu parametros.h
	nvcc -std=c++11 -O2 -o  $@ $< -Wno-deprecated-gpu-targets $(RANDOM123) --expt-extended-lambda

clean:
	rm -f $(APP)

run: all	
	./a.out 

submit:	all
	qsub jobGPU; watch qstat


