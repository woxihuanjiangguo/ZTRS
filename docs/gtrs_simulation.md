# Simulation
This document introduces how to acquire the trajectory statistics for training GTRS. This process consumes a considerable amount of CPU time and memory.

For large vocabularies like [V_16384](traj_final/16384.npy), we suggest using the pkls we provided [here](https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl).

Here are the detailed instructions for running your own simulations:
## Metric Caching
Before simulation, run metric caching for the Navtrain data split:
```shell
TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/navtrain_metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
```

## Splitting Navtrain
This step is to parallelize the simulation process for efficiency. We recommend splitting the Navtrain split into 32 subsets and run simulation on 32 separate machines:
```shell
python navsim/agents/tools/split_yamls.py
```
This generates 32 subsets of Navtrain under [navsim/planning/script/config/common/train_test_split](navsim/planning/script/config/common/train_test_split).

## Simulation
On each machine run the following script:
```shell
export split=navtrain
export part=1 # 1,2, ..., 32
export PROGRESS_MODE=gen_gt
export POSTFIX=v2

# threads_per_node should be tuned according to the machine's memory limit, if it is too large, ray_distributed will crash.
python $NAVSIM_DEVKIT_ROOT/navsim/agents/tools/gen_vocab_score.py \
train_test_split=${split}_${part} \
experiment_name=debug \
worker.threads_per_node=64 \
+save_name=${split}_${part} \
metric_cache_path=$NAVSIM_EXP_ROOT/${split}_metric_cache
```
It might take around 1-2 days on 32 separate machines to complete the simulation of V_16384 depending on your hardware.

## Merge Results
Merge the results from different machines with:

```shell
python navsim/agents/tools/merge_subsets.py
```