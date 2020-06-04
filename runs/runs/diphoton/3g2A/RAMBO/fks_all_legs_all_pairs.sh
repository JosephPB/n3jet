#!/bin/bash

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_10k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_10k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_10k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_20k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_20k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_20k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_30k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_30k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_30k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_40k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_40k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_40k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_50k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_50k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_50k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_60k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_60k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_60k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_70k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_70k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_70k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_80k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_80k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_80k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_90k.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_90k_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_90k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

wait

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_10k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_20k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_30k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_40k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_50k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_60k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_70k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_80k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &

python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_90k_fks_all_legs_all_pairs \
--training_reruns 20 \
--all_legs True \
--all_pairs True &
