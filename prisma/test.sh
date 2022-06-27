array=(
0.6
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
		--simTime=15 \
		--basePort=3000 \
		--train=0 \
		--seed=100 \
		--agent_type="sp" \
		--session_name="test_sp_load_$res1" \
		--link_delay="0ms" \
		--signaling_type="ideal" \
		--signalingSim=1 \
		--logs_parent_folder=examples/abilene/ \
		--traffic_matrix_index=0 \
		--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
		--node_coordinates_path=examples/abilene/node_coordinates.txt \
		--max_out_buffer_size=16260 \
		--save_models=0 \
		--start_tensorboard=0 \
		--load_factor=$j
done