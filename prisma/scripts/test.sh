#!/bin/bash  

#Testing DQN-Routing
cd .. 

#Load factors array. The execution will iterate over the array.
array=(
0.5
1
1.5
2
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

	python3 main.py --simTime=60 \
		--basePort=$((5655 + (counter*50) )) \
		--train=0 \
		--session_name="test_train_abilene_ideal_ts_0_03_batch_512_lr_1e-3_gamma_0_95_final_eps_0_01_ratio_0_1_loss_x6_load_$res1"\
		--logs_parent_folder=examples/abilene/ \
		--traffic_matrix_path=examples/abilene/traffic_matrices/node_intensity_normalized.txt \
		--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
		--node_coordinates_path=examples/abilene/node_coordinates.txt \
		--load_path=examples/abilene/saved_models/train_abilene_ideal_ts_0_03_batch_512_lr_1e-3_gamma_0_95_final_eps_0_01_ratio_0_1_loss_x6/iteration1_episode1 \
		--save_models=0 \
		--start_tensorboard=0 \
		--load_factor=$j
	counter=$((counter+1))
	echo $counter
	sleep 5
done





rm -r ../ns3-gym/scratch/prisma
