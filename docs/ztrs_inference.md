# Inference

## Format checkpoints
Before evaluation, the checkpoints in the training directory should be renamed. This step can be skipped if you use our provided checkpoints.

```bash
dir=your_path_to_exp # e.g. train_dp
cd ${NAVSIM_EXP_ROOT}/${dir}
for file in epoch=*-step=*.ckpt; do
    epoch=$(echo $file | sed -n 's/.*epoch=\([0-9][0-9]\).*/\1/p')
    new_filename="epoch${epoch}.ckpt"
    mv "$file" "$new_filename"
done
```

## ZTRS Inference
```bash
export PROGRESS_MODE="eval"
split=navhard
agent=ztrs_vov
dir=train_ztrs_vov
metric_cache_path="${NAVSIM_EXP_ROOT}/${split}_two_stage_metric_cache"
cd ${NAVSIM_DEVKIT_ROOT}

# we select the best-performing checkpoint in the last few epochs
for epoch in {10..14}; do
    padded_epoch=$(printf "%02d" $epoch)
    experiment_name="${dir}/test-${padded_epoch}ep-${split}-random"
    ckpt=${NAVSIM_EXP_ROOT}/${dir}/epoch${padded_epoch}.ckpt # this can also be the checkpoint we provided
    
    export DP_PREDS=None
    export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${dir}/epoch${epoch}_${split}.pkl; # save path for the scores

    python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
        agent=$agent \
        +combined_inference=false \
        dataloader.params.batch_size=32 \
        agent.checkpoint_path=${ckpt} \
        agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
        trainer.params.precision=32 \
        experiment_name=${experiment_name} \
        +cache_path=null \
        metric_cache_path=${metric_cache_path} \
        train_test_split=${split}_two_stage
done
```
