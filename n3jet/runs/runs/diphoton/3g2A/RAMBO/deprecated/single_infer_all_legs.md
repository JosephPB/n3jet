python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_1M_loop.npy \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_100k_single_all_legs_all_save \
--training_reruns 20 \
--all_legs True