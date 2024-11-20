C = g++ -std=c++17
ifeq ($(version), debug)
	C += -g -Wall -Wextra -Wconversion -Wunreachable-code -I.
else
	C += -O2 -I.
endif

all: main clean
main: main.o ml_sa.o nested_sa.o sa.o swap_model.o helpers.o
	$C -o main main.o ml_sa.o nested_sa.o sa.o swap_model.o helpers.o
main.o: main.cpp helpers.h ml_sa.h nested_sa.h sa.h swap_model.h
	$C -c main.cpp
swap_model.o: swap_model.cpp swap_model.h ml_sa.h nested_sa.h sa.h helpers.h
	$C -c swap_model.cpp
ml_sa.o: ml_sa.cpp ml_sa.h nested_sa.h sa.h helpers.h
	$C -c ml_sa.cpp
nested_sa.o: nested_sa.cpp nested_sa.h helpers.h sa.h
	$C -c nested_sa.cpp
sa.o: sa.cpp sa.h helpers.h
	$C -c sa.cpp
helpers.o: helpers.cpp helpers.h
	$C -c helpers.cpp
clean:
	mv main ../bin
	rm -f *o
