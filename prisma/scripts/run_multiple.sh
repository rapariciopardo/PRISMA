#!/bin/bash
./scripts/run_4_nodes.sh 1 100 0 dqn_buffer NN 100 10000 1 abilene 0 0.6 1 1000 dqn_buffer_4n compare_dqn_buffer_vs_q_routing_ &
./scripts/run_4_nodes.sh 1 100 0 dqn_routing NN 100 10000 2 abilene 0 0.6 1 1000 dqn_routing_4n compare_dqn_buffer_vs_q_routing_ &
./scripts/run_4_nodes.sh 1 100 0 sp ideal 100 10000 3 abilene 0 0.6 1 1000 sp_4n compare_dqn_buffer_vs_q_routing_ & 
./scripts/run_4_nodes.sh 1 100 0 opt ideal 100 10000 4 abilene 0 0.6 1 1000 opt_4n compare_dqn_buffer_vs_q_routing_ & 
./scripts/run_5_nodes.sh 1 100 0 dqn_buffer NN 100 10000 5 abilene 0 0.6 1 1000 dqn_buffer_5n compare_dqn_buffer_vs_q_routing_ &
./scripts/run_5_nodes.sh 1 100 0 dqn_routing NN 100 10000 6 abilene 0 0.6 1 1000 dqn_routing_5n compare_dqn_buffer_vs_q_routing_ &
./scripts/run_5_nodes.sh 1 100 0 sp ideal 100 10000 7 abilene 0 0.6 1 1000 sp_5n compare_dqn_buffer_vs_q_routing_ & 
./scripts/run_5_nodes.sh 1 100 0 opt ideal 100 10000 8 abilene 0 0.6 1 1000 opt_5n compare_dqn_buffer_vs_q_routing_ &