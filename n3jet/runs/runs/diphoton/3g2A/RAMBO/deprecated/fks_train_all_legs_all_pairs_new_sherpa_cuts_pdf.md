python general_init_model_rerun.py \
--mom_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_100k_new_sherpa_cuts_PDF.npy \
--nj_file /mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_100k_new_sherpa_cuts_PDF_loop.npy \
--delta_cut 0. \
--delta_near 0.02 \
--model_base_dir /scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/ \
--model_dir events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_overhaul \
--training_reruns 20 \
--all_legs True \
--all_pairs True
