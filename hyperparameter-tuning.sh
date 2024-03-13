# for config_file in 'train_baseline_local_models.yaml' 'train_pretrained_xception_model.yaml' 'train_randominit_xception_model_no_latlons_train_all.yaml' 'train_randominit_xception_model_no_latlons.yaml' 'train_randominit_xception_model.yaml'; do
for config_file in 'train_baseline_local_models.yaml'; do
    for lr in 0.00001 0.0001 0.001 0.01; do
        for weight_decay in 0.00001 0.0001 0.001 0.01; do
            sbatch job.sh ${config_file} ${lr} ${weight_decay}
        done
    done
done