#!/bin/bash  

#Testing DQN-Routing
cd .. 

#Load factors array. The execution will iterate over the array.
array=(
0.6
0.7
0.8
0.9
1.0
1.1
1.2
1.3
1.4
)
counter=0

#For running different agents, add the following arg:
# --agent_type=sp \   #e.g., for Shortest Path agent
for j in ${array[@]}
	do 
	echo $j
	
    FLOAT=$(echo $j*10000 | bc)
    res1=${FLOAT/.*}
	echo $res1

	python3 main.py --simTime=15 \
		--basePort=$((7655 + (counter*50) )) \
		--train=0 \
		--session_name="test_abilene_train_tiago_v29_load_$res1"\
		--link_delay="0ms" \
		--signaling_type="ideal" \
		--logs_parent_folder=examples/abilene/ \
		--traffic_matrix_index=3 \
		--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
		--node_coordinates_path=examples/abilene/node_coordinates.txt \
		--load_path=examples/abilene/saved_models/abilene_train_tiago_v29/iteration1_episode1 \
		--save_models=0 \
		--start_tensorboard=0 \
		--load_factor=$j
	counter=$((counter+1))
	echo $counter
	sleep 5
done





#rm -r ../ns3-gym/scratch/prisma
