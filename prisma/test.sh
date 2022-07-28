array=(
1.0
#0.8
#1.0
#1.2
#1.4
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
		--simTime=500 \
		--basePort=3000 \
		--train=1 \
		--seed=100 \
		--agent_type="dqn_buffer" \
		--session_name="test_tiago_v32_0_5_10_7_v5_load_$res1" \
		--link_delay="0ms" \
		--signaling_type="NN" \
		--signalingSim=1 \
		--training_step=0.01 \
		--batch_size=256 \
		--lr=0.001 \
		--gamma=1.0 \
		--exploration_final_eps=0.01 \
		--exploration_initial_eps=1.0 \
		--iterationNum=10000 \
		--training_trigger_type="time" \
		--save_models=0 \
		--start_tensorboard=0 \
		--replay_buffer_max_size=15000 \
   		--link_delay="1ms" \
		--load_factor=$j \
		--logs_parent_folder=examples/abilene/ \
		--traffic_matrix_index=5 \
		--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
		--node_coordinates_path=examples/abilene/node_coordinates.txt \
		--max_out_buffer_size=16260 \
		#--load_path=examples/abilene/dqn_buffer_sp_init
done