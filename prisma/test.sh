array=(
0.4
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
		--simTime=30 \
		--basePort=3000 \
		--train=1 \
		--seed=100 \
		--agent_type="dqn_buffer" \
		--session_name="test_tiago_load_$res1" \
		--link_delay="0ms" \
		--signaling_type="NN" \
		--signalingSim=1 \
		--logs_parent_folder=examples/abilene/ \
		--traffic_matrix_index=0 \
		--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
		--node_coordinates_path=examples/abilene/node_coordinates.txt \
		--max_out_buffer_size=16260 \
		--save_models=1 \
		--start_tensorboard=0 \
		--load_factor=$j
done