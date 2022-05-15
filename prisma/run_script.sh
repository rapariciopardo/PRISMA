 # --load_path="examples/abilene/saved_models/abilene_train_BS_512_LR_00001_LD04_noprop/iteration1_episode1" \
python3 main.py \
   --seed=3 \
   --basePort=8755 \
   --train=1 \
   --session_name="abilene_train_tiago_v33" \
   --logs_parent_folder="examples/abilene/" \
   --traffic_matrix_index=0 \
   --adjacency_matrix_path="examples/abilene/adjacency_matrix.txt" \
   --node_coordinates_path="examples/abilene/node_coordinates.txt" \
   --load_factor=0.6 \
   --start_tensorboard=0 \
   --agent_type="dqn_buffer" \
   --simTime=5 \
   --signaling_type="ideal" \
   --save_models=1 \
   --exploration_initial_eps=1.0 \
   --exploration_final_eps=0.01 \
   --iterationNum=3000 \
   --batch_size=512 \
   --sync_ratio=0.2 \
   --sync_step=0.5 \
   --training_step=0.01 \
   --lr=0.001 \
   --replay_buffer_max_size=50000 \
   --link_delay="0ms" \
   --tensorboard_port=65534 \
