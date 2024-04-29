for lr in 0.00001 0.0001 0.001 0.01; do
    for weight_decay in 0.00001 0.0001 0.001 0.01; do
        sbatch job.sh 'train_pretrained_xception_model.yaml' ${lr} ${weight_decay} 9
    done
done