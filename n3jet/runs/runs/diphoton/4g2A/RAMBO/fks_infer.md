python general_precision_error_testing.py \
--test_mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/4g2A/RAMBO/momenta_events_1M.npy \
--test_nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/4g2A/RAMBO/events_1M_loop.npy \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/4g2A/RAMBO/ \
--model_dir events_100k_fks \
--training_reruns 20