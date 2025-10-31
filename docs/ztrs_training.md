# Training

Before training, make sure that the pretrained vision backbones and simulated ground-truths for trajectory scorers are downloaded according to [install.md](install.md).

We provide the training script utilizing three machine nodes (each with 8 NVIDIA A00 GPUS). 
If you plan to train on a single-node machine, set the variable **NUM_NODES** to 1. 
To ensure a comparable number of gradient updates, set **max_epochs** to 5.

Here is the training script with 3 nodes:
```bash
NUM_NODES=3
MASTER_ADDR=MASTER_NODE_IP # your master node ip, which can be set to 127.0.0.1 for single-node training 
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
experiment_name=train_ztrs_vov # this could also be train_hydra_mdp
agent=ztrs_vov # the agent could also be hydra_mdp_vov

# training hyper-parameters
lr=0.0002
bs=22
max_epochs=15

MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ztrs.py \
            --config-name ${config} \
            trainer.params.num_nodes=${NUM_NODES} \
            agent=${agent} \
            experiment_name=${experiment_name} \
            train_test_split=navtrain \
            dataloader.params.batch_size=${bs} \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=32 \
            agent.config.ckpt_path=${experiment_name} \
            agent.lr=${lr} \
            cache_path=null
```