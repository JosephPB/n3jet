CXX?=g++
CXXFLAGS=-g -std=c++11 -Wall -Wextra -Wpedantic
DEFS=-DDEBUG 

.PHONY: clean all

all: ex_3g2A single_test

# Example of using the NN
ex_3g2A_multiple_fks: ex_3g2A_multiple_fks.o model_fns.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

ex_3g2A_multiple_single: ex_3g2A_multiple_single.o model_fns.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

ex_3g2A_test: ex_3g2A_test.o model_fns.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

single_test: single_test.o model_fns.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

model_fns.o: model_fns.cpp model_fns.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

ex_3g2A_test.o: ex_3g2A_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

single_test.o: single_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS) $(DEFS)

ex_3g2A_multiple_single.o: ex_3g2A_multiple_single.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

ex_3g2A_multiple_fks.o: ex_3g2A_multiple_fks.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CPPFLAGS)

clean:
	-rm -f *.o __pycache__ ex_3g2A_multiple_fks ex_3g2A_multiple_single ex_3g2A_test single_test
