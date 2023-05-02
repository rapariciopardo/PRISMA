#!/bin/bash
# ./scripts/run_4_nodes copy.sh 1 100 0 dqn_buffer_with_throughputs NN 10 10000 10 abilene 0 0.6 1 1000 a test 5 
# ./scripts/run_4_nodes.sh 1 100 0 dqn_buffer NN 10 10000 12 abilene 0 0.6 1 1000 dqn_buffer_10_fr10 dqn_buffer_4n_moving_avg_variation 10 &
./scripts/run_4_nodes.sh 1 100 0 dqn_buffer NN 10 10000 2 abilene 0 0.6 1 500 dqn_buffer_10_fr10_explo_10 dqn_buffer_4n_mesh_moving_avg_variation 10 &
./scripts/run_4_nodes.sh 1 100 0 sp ideal 10 10000 3 abilene 0 0.6 1 500 sp dqn_buffer_4n_mesh_moving_avg_variation 10 &
./scripts/run_4_nodes.sh 1 100 0 opt ideal 10 10000 4 abilene 0 0.6 1 500 opt dqn_buffer_4n_mesh_moving_avg_variation 10 &

# ./scripts/run_11_nodes.sh 1 100 0 dqn_buffer NN 10 10000 7 abilene 0 0.6 1 60 test_11_mv_avg_5 11Nodes 5 1 &
# ./scripts/run_11_nodes.sh 1 100 0 dqn_buffer NN 10 10000 2 abilene 0 0.6 1 60 test_11_mv_avg_1 11Nodes 1 1 &
# ./scripts/run_11_nodes.sh 1 100 0 dqn_buffer NN 10 10000 3 abilene 0 0.6 1 60 test_11_mv_avg_20 11Nodes 20 1 &

# ./scripts/run_11_nodes.sh 1 100 0 dqn_buffer NN 10 10000 4 abilene 0 0.6 1 60 test_11_buffer 11Nodes 5 0 &
# ./scripts/run_11_nodes.sh 1 100 0 sp ideal 10 10000 5 abilene 0 0.6 1 60 test_11_buffer 11Nodes 5 0 &
# ./scripts/run_11_nodes.sh 1 100 0 opt ideal 10 10000 6 abilene 0 0.6 1 60 test_11_buffer 11Nodes 5 0 &

./scripts/run_5_nodes.sh 1 100 0 dqn_buffer_with_throughputs NN 10 10000 10 abilene 0 0.6 1 200 dqn_buffer_10_explo_10_fr10_throughputs dqn_buffer_5n_moving_avg_variation_fixed 10 &
# ./scripts/run_5_nodes.sh 1 100 0 dqn_buffer NN 10 10000 11 abilene 0 0.6 1 200 dqn_buffer_100_explo_10_fr10 dqn_buffer_5n_moving_avg_variation 100 &
# ./scripts/run_5_nodes.sh 1 100 0 sp ideal 100 10000 9 abilene 0 0.6 1 1000 sp dqn_buffer_5n_moving_avg_variation 100 &
# ./scripts/run_5_nodes.sh 1 100 0 opt ideal 100 10000 10 abilene 0 0.6 1 1000 opt dqn_buffer_5n_moving_avg_variation 100 &
