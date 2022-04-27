#!/bin/bash  

## Training DQN

#For running different agents, add the following arg:
# --agent_type=sp \   #e.g., for Shortest Path agent

cd ..

python3 main.py \
	--seed=100 \
	--simTime=60 \
	--basePort=6555 \
	--train=1 \
	--agent_type="dqn_buffer" \
	--session_name="train_abilene_ideal_ts_0_03_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_sync_05_loss_x1_sp_init"\
	--signaling_type="ideal" \
	--logs_parent_folder=examples/abilene/ \
	--traffic_matrix_root_path=examples/abilene/traffic_matrices/ \
	--traffic_matrix_index=0 \
	--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
	--node_coordinates_path=examples/abilene/node_coordinates.txt \
	--training_step=0.01 \
	--batch_size=512 \
	--lr=0.001 \
	--exploration_final_eps=0.01 \
	--exploration_initial_eps=1.0 \
	--iterationNum=3000 \
	--gamma=1.0 \
	--training_trigger_type="time" \
	--save_models=1 \
	--start_tensorboard=0 \
	--replay_buffer_max_size=50000 \
   	--link_delay="0ms" \
	--load_factor=0.4 \
	--sync_step=0.5 \
	--load_path=examples/abilene/DQN_buffer_sp_init \

sleep 5

rm -r ../ns3-gym/scratch/prisma
