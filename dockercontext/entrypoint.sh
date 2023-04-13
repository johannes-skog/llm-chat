#!/usr/bin/bash
screen -S notebook -dm bash -c "jupyter notebook --ip 0.0.0.0 --allow-root --port 8080"
screen -S tensorboard -dm bash -c "tensorboard --logdir artifacts --port 6006 --host 0.0.0.0"

PATH=/opt/conda/bin/:${PATH}

HF_DATASETS_CACHE="/hgcache"

sleep infinity