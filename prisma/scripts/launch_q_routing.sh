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


## Training DQN
## If using, uncomment the lines below

python3 main.py --simTime=5 \
	--basePort=6855 \
	--train=1 \
	--session_name="q_routing_no_signaling_time"\
	--logs_parent_folder=examples/abilene/ \
	--traffic_matrix_path=examples/abilene/traffic_matrices/node_intensity_0.txt \
	--training_step=0.1 \
	--training_trigger_type="time" \
	--save_models=0 \
	--start_tensorboard=0 \
	--tensorboard_port=16666


## Testing dqn
## Uncoment the lines below 
#couter=0
#batchSize=10
#for ((i=0; i<4; i++))
#	do  for ((j=100; j<=500; j=j+100))
#		do 
#		res=$((counter%batchSize))
#		if (($res==0)); then
#			wait
#		fi
#		sleep $i
#		python3 multi_agents_threaded.py --simTime=60 \
#			--basePort=$((6555+$j + (15*$i))) \
#			--session_name="dqn_mat_$(($i))_seed_$j" \
#			--logs_parent_folder=outputs/test_dqn_buffer_no_signaling/ \
#			--seed=100 \
#			--start_tensorboard=0 \
#			--train=0 \
#			--max_nb_arrived_pkts=10000 \
#			--load_path=saved_models/dqn_mat_$(($i))_seed_$j/episode1 \
#			--save_models=0 \
#			--exploration_initial_eps=0 \
#			--exploration_final_eps=0 \
#			--load_factor=0.01&
#		counter=$((counter+"1"))
#	done
#done


rm -r ../ns3-gym/scratch/prisma
