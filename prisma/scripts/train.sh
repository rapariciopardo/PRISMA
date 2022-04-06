#!/bin/bash  

## Training DQN

#For running different agents, add the following arg:
# --agent_type=sp \   #e.g., for Shortest Path agent

cd ..

python3 main.py --simTime=60 \
	--basePort=6555 \
	--train=1 \
	--session_name="train_abilene_ideal_ts_0_03_batch_512_lr_1e-3_gamma_0_95_final_eps_0_01_ratio_0_1_loss_x6"\
	--signaling_type="ideal" \
	--logs_parent_folder=examples/abilene/ \
	--traffic_matrix_path=examples/abilene/traffic_matrices/node_intensity_normalized.txt \
	--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
	--node_coordinates_path=examples/abilene/node_coordinates.txt \
	--training_step=0.030 \
	--batch_size=512 \
	--lr=0.001 \
	--exploration_final_eps=0.01 \
	--exploration_initial_eps=1.0 \
	--iterationNum=3000 \
	--gamma=1.0 \
	--training_trigger_type="time" \
	--save_models=1 \
	--start_tensorboard=0 \
	--load_factor=0.5
sleep 5

rm -r ../ns3-gym/scratch/prisma
