for lr in 0.00001 0.0001 0.001 0.01; do
    for weight_decay in 0.00001 0.0001 0.001 0.01; do
        for channels in 3 4 12 15; do
            sbatch hyperparameter_tuning.sh 'train_baseline_local_models.yaml' ${lr} ${weight_decay} --channels=${channels} 
        done
        for layers_tuned in 1 2 3; do
            sbatch hyperparameter_tuning.sh 'train_pretrained_xception_model.yaml' ${lr} ${weight_decay} --layers_tuned=${layers_tuned} 
        done
        for layers_tuned in 1 2 3; do
            sbatch hyperparameter_tuning.sh 'train_randominit_xception_model_no_latlons.yaml' ${lr} ${weight_decay} --layers_tuned=${layers_tuned} 
        done
        for layers_tuned in 1 2 3; do
            sbatch hyperparameter_tuning.sh 'train_randominit_xception_model_latlons.yaml' ${lr} ${weight_decay} --layers_tuned=${layers_tuned} 
        done
        sbatch hyperparameter_tuning.sh 'train_unet_randominit.yaml' ${lr} ${weight_decay} --channels=12
        for freeze_backbone in 'True' 'False'; do
            sbatch hyperparameter_tuning.sh 'train_unet_ssl4eoweights.yaml' ${lr} ${weight_decay} --freeze_backbone=${freeze_backbone} 
        done
    done
done