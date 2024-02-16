# for lr in 0.001; do
#     for weight_decay in 0.00001; do
for lr in 0.00001 0.0001 0.001 0.01 0.1; do
    for weight_decay in 0.00001 0.0001 0.001 0.01 0.1; do
        sbatch job.sh ${lr} ${weight_decay}
    done
done