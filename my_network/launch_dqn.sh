#!/bin/bash  
## for training dqn
cp -r ../my_network ../ns3-gym/scratch
cd ../ns3-gym
./waf configure && ./waf build
sleep 3
cd ../my_network


## Training DQN
## If using, uncomment the lines below
for ((i=0; i<1; i++))
	do  for ((j=100; j<=100; j=j+100))
		do 
		python3 multi_agents_threaded.py --simTime=40 \
			--basePort=$((6555+$j + (15*$i))) \
			--train=1 \
			--session_name="q_routing_mat_$(($i))_seed_$j"\
			--logs_parent_folder=examples/abilene/outputs/train_q_routing_no_signaling/ \
			--traffic_matrix_path=scratch/my_network/examples/abilene/traffic_matrices/node_intensity_$(($i)).txt \
			--seed=$j \
			--load_factor=0.01 \
			--start_tensorboard=0 &
		sleep 5
	done
done

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


rm -r ../ns3-gym/scratch/my_network
