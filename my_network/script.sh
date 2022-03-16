#!/bin/bash  
## for training dqn
for ((i=0; i<1; i++))
	do  for ((j=100; j<=100; j=j+100))
		do 
		python multi_agents_threaded.py --simTime=40 \
			--basePort=$((6555+$j + (15*$i))) \
			--train=1 \
			--session_name="q_routing_mat_$(($i))_seed_$j"\
			--logs_parent_folder=examples/abilene/ \
			--traffic_matrix_path=scratch/my_network/examples/abilene/traffic_matrices/node_intensity_$(($i)).txt \
			--seed=$j \
			--load_factor=0.01 \
			--start_tensorboard=0
		# sleep 5
	done
done


## for testing dqn 
# couter=0
# batchSize=10
# for ((i=0; i<4; i++))
# 	do  for ((j=100; j<=500; j=j+100))
# 		do 
# 		res=$((counter%batchSize))
# 		if (($res==0)); then
# 			wait
# 		fi
# 		sleep $i
# 		python multi_agents_threaded.py --simTime=60 \
# 			--basePort=$((6555+$j + (15*$i))) \
# 			--session_name="dqn_mat_$(($i))_seed_$j" \
# 			--logs_parent_folder=outputs/test_dqn_buffer_no_signaling/ \
# 			--seed=100 \
# 			--start_tensorboard=0 \
# 			--train=0 \
# 			--max_nb_arrived_pkts=10000 \
# 			--load_path=saved_models/dqn_mat_$(($i))_seed_$j/episode1 \
# 			--save_models=0 \
# 			--exploration_initial_eps=0 \
# 			--exploration_final_eps=0 \
# 			--load_factor=0.01&
# 		counter=$((counter+"1"))

# 	done
# done


## test SP with load factor variation
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
# for ((i=0; i<4; i++))
# 	do  for j in ${array[@]}
# 		do 
# 		res=$((counter%batchSize))
# 		if (($res==0)); then
# 			wait
# 		fi
# 		sleep $i
# 		res1=$(printf "%.0f" $(echo "scale=2; ($j  / 0.01 * 100 )" | bc ))
# 		echo $res1
# 		python multi_agents_threaded.py --simTime=600 \
# 			--basePort=$((6555 + (15*counter))) \
# 			--session_name="sp_mat_$(($i))_load_$j" \
# 			--logs_parent_folder=outputs/tests/mat_$(($i))/load_$(($res1))/sp/ \
# 			--seed=100 \
# 			--start_tensorboard=0 \
# 			--train=0 \
# 			--agent_type=sp \
# 			--max_nb_arrived_pkts=10000 \
# 			--save_models=0 \
# 			--exploration_initial_eps=0 \
# 			--exploration_final_eps=0 \
# 			--traffic_matrix_path=scratch/my_network/traffic_matrices/node_intensity_$(($i)).txt \
# 			--load_factor=$j &		
# 		counter=$((counter+"1"))

# 	done
# done


## test DQN with load factor variation
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
# 				--traffic_matrix_path=scratch/my_network/traffic_matrices/node_intensity_$(($i)).txt \
# 				--load_factor=$j &		
# 			counter=$((counter+"1"))
# 		done
# 	done
# done