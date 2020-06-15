# Converting and calling models in C++

Converting a keras trained model into a format callable by C++ operations including data loading.
This code is based on and adapted from: [keras2cpp](https://github.com/pplonski/keras2cpp)


## End to end single model example

This example will train a simple model and save the output in an architecture `.json` file and `.h5` weights file. These files will be used to convert the model into a `.nnet ` file which can be read in by the functions in `model_fns.cpp`.

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
python conversion/dump_to_simple_cpp.py -a tests/single_test_arch.json -w /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/single_test_weights.h5 -o tests/single_test_dumped.nnet
```

Compile:
```
make single_test
```

Run:
```
./single_test
```

## 3g2A example

This example will infer on a model once trained to predict the loop squared of a $gg /to g /gamma /gamma$ process. Here we read in some momenta stored in the normal `double` array format `double Momenta[pspoints][legs][4]` and go through the following procedure for each phase-space point:

1. Flatten the array
2. Standardise each element in the array based on the mean and standard deviations over the whole dataset (see the `NN.process_training_data()` in the `tests/single_test_train.py` script)
3. Infer on the loaded neural network
4. Destandardise the output and print

**Note:** There needs to be an already trained network to do this saved in `tests/single_test_dumped.nnet`. To get this you can either use the `.nnet` file provided or run the example above which will generate one.

Compile:
```
make ex_3g2A
```

Run:
```
./ex_3g2A
```
