#!/bin/bash 

## Move to main folder
cd ..

## Copy prisma into ns-3 folder
rsync -r --exclude-from=../.gitignore ../prisma ../ns3-gym/scratch/

## configure ns3
cd ../ns3-gym
./waf -d optimized configure
sleep 3
cd ../prisma

## Test SP with load factor variation
## Uncomment the lines below
array=(
0.039
0.04
0.05
)
counter=0
#batchSize=10
for ((i=4; i<5; i++))
	do  for j in ${array[@]}
		do 
		echo $j
		#res=$((counter%batchSize))
		#if (($res==0)); then
		#	wait
		#fi
		sleep $i
        FLOAT=$(echo $j*10000 | bc)
        res1=${FLOAT/.*}
		echo $res1
		python3 multi_agents_threaded.py --simTime=600 \
			--basePort=$((6555 + (counter*50) )) \
			--session_name="sp_mat_$(($i))_load_$res1" \
			--logs_parent_folder=examples/abilene/outputs/mat_$(($i))/load_$(($res1))/sp/ \
			--seed=100 \
			--start_tensorboard=0 \
			--train=0 \
			--agent_type=sp \
			--max_nb_arrived_pkts=10000 \
			--save_models=0 \
			--exploration_initial_eps=0 \
			--exploration_final_eps=0 \
			--traffic_matrix_path=scratch/prisma/examples/abilene/traffic_matrices/node_intensity_$(($i)).txt \
			--adjacency_matrix_path=scratch/prisma/examples/abilene/adjacency_matrix.txt \
			--node_coordinates_path=scratch/prisma/examples/abilene/node_coordinates.txt \
			--load_factor=$j 		
		counter=$((counter+1))
		echo $counter
        sleep 3
	done
done


## test DQN with load factor variation
## Uncomment the lines below
# array=(
# 0.0005
# 0.001
# 0.005
# 0.01
# 0.015
# 0.02
# 0.025
# 0.03
# )
# counter=0
# batchSize=10
# for ((i=0; i<1; i++))
# do  for j in ${array[@]}
# 	do for ((k=100; k<=500; k=k+100))
# 		do 
# 			res=$((counter%batchSize))
# 			if (($res==0)); then
# 				wait
# 			fi
# 			sleep $i
# 			res1=$(printf "%.0f" $(echo "scale=2; ($j  / 0.01 * 100 )" | bc ))
# 			echo $res1
# 			python multi_agents_threaded.py --simTime=600 \
# 				--basePort=$((6555 + (15*counter))) \
# 				--session_name="dqn_mat_$(($i))_load_$j" \
# 				--logs_parent_folder=outputs/tests/mat_$(($i))/load_$(($res1))/dqn_buffer/seed_$(($k))/ \
# 				--seed=100 \
# 				--start_tensorboard=0 \
# 				--train=0 \
# 				--agent_type=dqn \
# 				--max_nb_arrived_pkts=10000 \
# 				--save_models=0 \
# 				--exploration_initial_eps=0 \
# 				--exploration_final_eps=0 \
# 				--load_path=saved_models/dqn_mat_$(($i))_seed_$k/episode1 \
# 				--traffic_matrix_path=scratch/prisma/examples/abilene/traffic_matrices/node_intensity_$(($i)).txt \
# 				--load_factor=$j &		
# 			counter=$((counter+"1"))
# 		done
# 	done
# done

rm -r ../ns3-gym/scratch/prisma
