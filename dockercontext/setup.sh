#!/usr/bin/bash
apt-get -y install git
apt-get install screen -y
pip install -r requirements.txt
#screen -S tensorboard -dm bash -c "tensorboard --logdir artifacts --port 6006"
