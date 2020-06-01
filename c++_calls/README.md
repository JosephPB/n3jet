# Converting and calling models in C++

Converting a keras trained model into a format callable by C++ operations including data loading.
This code is based on and adapted from: [keras2cpp](https://github.com/pplonski/keras2cpp)


## Single model test

Current procedure:

Run model training:
```
cd tests/
python single_test_infer.py
```

Dump network:
```
cd ..
python dump_to_simple_cpp.py -a tests/single_test_arch.json -w /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/single_test_weights.h5 -o tests/single_test_dumped.nnet
```

Compile:
```
g++ -std=c++11 model_fns.cpp single_main.cpp
```

Run:
```
./a.out
```