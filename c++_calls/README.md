# Converting and calling models in C++

Converting a keras trained model into a format callable by C++ operations including data loading.
This code is based on and adapted from: [keras2cpp](https://github.com/pplonski/keras2cpp)


## Single model test

This test will train a simple model and save the output in an architecture `.json` file and `.h5` weights file. These files will be used to convert the model into a `.nnet ` file which can be read in by the functions in `model_fns.cpp`.

**Note:**
- Here we assume the input data is already processed as required for the model (i.e. mean subtracted and normalised to the standard deviation as performed by the `process_training_data()` function in `n3jet/models/model.py`)
- A weights output file must be specified - for running on the IPPP system this should be to `/scratch/` for better read/write time

Run model training:
```
cd tests/
python single_test_train.py -w /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/single_test_weights.h5
```

Dump network:
```
cd ..
python dump_to_simple_cpp.py -a tests/single_test_arch.json -w /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/single_test_weights.h5 -o tests/single_test_dumped.nnet
```

Compile:
```
g++ -std=c++11 model_fns.cpp single_test.cpp -o single_test.o
```

Run:
```
./single_test.out
```