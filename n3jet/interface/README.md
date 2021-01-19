# NJet Interface

This directory contains files for running NJet based on order/contract files and additional flag options that can be added at call time.

## TL;DR

- Edit `global_dict.py` as necessary
- Run `python global_dict.py`
- Run `python interface.py --[flags]`

## Dictionary

The `global_dict.py` script must be edited and run before running the interface. This file points to your local njet repositories and library directories. This makes it easy to manage njet versions.

## Contract processing

If a contract file does not exist but an order file does (following BLHA guidelines) then this can be created. the `contract.py` script can do this and just calls the `njet.py` script residing in `[NJET_DIRECTORY]/blha/njet.py`.

**Note:** The order file does not need to be processed in advance if using the interface since you can just pass the flag `--proc_order True` to achieve this (see next section)

## Interface

The interface is called by running `interface.py`. This has numerous flags that can be determined by running

```python interface.py --help```

The possible other flags are:

```

  --order ORDER         name of order file, default: None
  --contract CONTRACT   name of contract file, default: None
  --proc_order PROC_ORDER
                        process order file to create contract file and save in
                        same directory as order file, default: False
  --nlegs NLEGS         number of external legs - only needed if not giving a
                        mom file
  --nps NPS             number of phase-space points
  --mom_file MOM_FILE   destination of momenta file
  --generate_mom GENERATE_MOM
                        generate momenta file even if it already exists,
                        default: False
  --nj_file NJ_FILE     NJet file
  --generate_nj GENERATE_NJ
                        generate NJet file even if it already exists, default:
                        False
  --amp_type AMP_TYPE   amplitude type - set: tree, loop, loopsq (Note:
                        6-point diphoton uses loopsq)
  --mur MUR             renormalisation scale - currently this can not by
                        dynamic, default: 91.188
  --alpha ALPHA         alpha parameter value, default: 1. (Note: 6-point
                        diphoton uses 1./137.035999084)
  --alphas ALPHAS       alphas parameter value, default: 1. (Note: e+e- uses
                        0.07957747155, 6-point diphoton uses 0.118)
  --blha BLHA           Use BLHA1 or BLHA2, set 1 or 2 - affects
                        EvalSubProcess type used, default: 2
  --debug DEBUG         in debug mode, value of first momenta value is printed
                        out, set True or False, default: False
```

An example of how this might be run the e+e- -> qqg at 1-loop for generating 100k points and processing the order file can be found here:

```
python interface.py \
--order /mt/home/jbullock/n3jet_runs/runs/e+e-_qq_plus_jets/qqg_V/RAMBO/OLE_order.lh \
--contract /mt/home/jbullock/n3jet_runs/runs/e+e-_qq_plus_jets/qqg_V/RAMBO/OLE_contract.lh \
--proc_order True \
--nlegs 5 \
--nps 100000 \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/e+e-_qq_plus_jets/qqg_V/RAMBO/momenta_events_100k.npy \
--generate_mom False \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/e+e-_qq_plus_jets/qqg_V/RAMBO/events_100k.npy \
--generate_nj True \
--amp_type loop \
--mur 91.188 \
--alpha 1. \
--alphas 0.07957747155 \
--blha 1 \
--debug False
```

**Note:** The interface works with BLHA 1 and BLHA 2 formatting guidelines. Some processes in NJet might only be written to comply with BLHA 2.

**Note:** This interface fully supports diphoton processes found in the NJet development version: `njet-develop`