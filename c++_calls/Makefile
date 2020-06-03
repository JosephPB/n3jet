CXX=g++
CXXFLAGS=-std=c++11
LDFLAGS=model_fns.cpp

COMPILE.cpp = $(CXX) $(CXXFLAGS) $(LDFLAGS)
OUTPUT_OPTION = -o $@

%.o: %.cpp
	$(COMPILE.cpp) $< $(OUTPUT_OPTION)

clean:
	rm -f *.0 __pycache__
