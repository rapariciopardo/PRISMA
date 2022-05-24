array=(
0.6
0.8
1.0
1.2
1.4
)
counter=0

#For running different agents, add the following arg:
# --agent_type=sp \   #e.g., for Shortest Path agent
for j in ${array[@]}
	do 
	echo $j
	FLOAT=$(echo $j*1000 | bc)
	res1=${FLOAT/.*}
	echo $res1
	python3 main.py \
		--simTime=15 \
		--basePort=3000 \
		--train=0 \
		--seed=100 \
		--session_name="test_abilene_sync_step_variation_mat_0_seed_100_load_$res1" \
		--link_delay="0ms" \
		--signaling_type="ideal" \
		--agent_type="dqn_buffer" \
		--logs_parent_folder=examples/abilene/ \
		--traffic_matrix_index=0 \
		--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
		--node_coordinates_path=examples/abilene/node_coordinates.txt \
		--load_path=examples/abilene/saved_models/train_abilene_ideal_ts_0_03_seed_100_traff_mat_0_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_sync_0.5_loss_x1_sp_init/iteration1_episode1 \
		--save_models=0 \
		--start_tensorboard=0 \
		--load_factor=$j
done