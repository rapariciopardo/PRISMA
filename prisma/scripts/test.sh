#!/bin/bash  
 
## Move to main folder
cd ..

## Copy prisma into ns-3 folder
rsync -r --exclude-from=../.gitignore ../prisma ../ns3-gym/scratch/

## configure ns3
cd ../ns3-gym
mv scratch/prisma/ns3/* scratch/prisma/.

./waf -d optimized configure
sleep 3
cd ../prisma


#Testing Q-Routing

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
		--session_name="test_load_$res1"\
		--logs_parent_folder=examples/geant/ \
		--traffic_matrix_path=examples/geant/traffic_matrices/node_intensity_normalized.txt \
		--adjacency_matrix_path=examples/geant/adjacency_matrix.txt \
		--node_coordinates_path=examples/geant/node_coordinates.txt \
		--load_path=examples/geant/saved_models/q_routing_geant_tiago_lr_1e-4_batch_512_iter_3000_eps_1_step_0_007_load_0_5/iteration1_episode1 \
		--save_models=0 \
		--start_tensorboard=0 \
		--load_factor=$j
	counter=$((counter+1))
	echo $counter
	sleep 5
done





rm -r ../ns3-gym/scratch/prisma
