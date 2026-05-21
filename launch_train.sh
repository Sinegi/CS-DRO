python train_all.py PACS0 \
        --algorithm CS_DRO \
        --data_dir ./dataset \
        --dataset PACS \
        --holdout_fraction 0.2 \
        --output_dir train_output/pacs \
        --seed 0 \
        --lambda_r 1.0 \
        --lambda_G 10 \
        --kappa 1.0 \
        --hp_tau 10.0 \
        --trial_seed 0 \
        --hparams '{"hidden_size": 256, "out_dim": 256, "checkpoint_freq": 200, "steps": 5000}'


python train_all.py VC0 \
        --algorithm CS-DRO \
        --data_dir ./dataset \
        --dataset VLCS \
        --holdout_fraction 0.2 \
        --output_dir train_output/vlcs \
        --seed 0 \
        --lambda_r 1.0 \
        --lambda_G 10 \
        --kappa 1.0 \
        --hp_tau 10.0 \
        --trial_seed 0 \
        --hparams '{"hidden_size": 512, "out_dim": 512, "checkpoint_freq": 200, "steps": 5000}'


python train_all.py OH0 \
        --algorithm CS-DRO \
        --data_dir ./dataset \
        --dataset OfficeHome \
        --holdout_fraction 0.2 \
        --output_dir train_output/officeHome \
        --seed 0 \
        --lambda_r 1.0 \
        --lambda_G 1 \
        --kappa 0.001 \
        --hp_tau 10.0 \
        --trial_seed 0 \
        --hparams '{"hidden_size": 512, "out_dim": 512, "checkpoint_freq": 200, "steps": 5000}'


python train_all.py TR0 \
        --algorithm CS-DRO \
        --data_dir ./dataset \
        --dataset TerraIncognita \
        --holdout_fraction 0.2 \
        --output_dir train_output/terraincognita \
        --seed 0 \
        --lambda_r 1.0 \
        --lambda_G 0.1 \
        --kappa 10.0 \
        --hp_tau 10.0 \
        --trial_seed 0 \
        --hparams '{"hidden_size": 512, "out_dim": 512, "checkpoint_freq": 200, "steps": 5000}'
