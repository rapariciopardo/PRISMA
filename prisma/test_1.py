import unittest
import subprocess

class TestMyProgram(unittest.TestCase):
    def test_output(self):
        command = 'sudo docker run --rm --gpus all -v /home/redha/PRISMA_copy:/prisma_ -w /prisma_/prisma allicheredha/prismacopy_episodes:offband_ python3 -u main.py --seed=100 --simTime=1 --train=0 --basePort=7000 --agent_type=dqn_buffer --signaling_type=ideal --logs_parent_folder=examples/5n_overlay_full_mesh_abilene/results/Compare_RCPO_offband --traffic_matrix_root_path=examples/5n_overlay_full_mesh_abilene/traffic_matrices/ --traffic_matrix_index=0 --overlay_adjacency_matrix_path=examples/5n_overlay_full_mesh_abilene/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/5n_overlay_full_mesh_abilene/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/5n_overlay_full_mesh_abilene/topology_files/node_coordinates.txt --map_overlay_path=examples/5n_overlay_full_mesh_abilene/topology_files/map_overlay.txt --training_step=0.01 --batch_size=512 --lr=1e-05 --exploration_final_eps=0.01 --exploration_initial_eps=1.0 --iterationNum=5000 --gamma=1.0 --save_models=0 --start_tensorboard=0 --replay_buffer_max_size=10000 --link_delay=1 --load_factor=0.1 --sync_step=1 --max_out_buffer_size=10000 --sync_ratio=0.2 --signalingSim=1 --movingAverageObsSize=5 --prioritizedReplayBuffer=0 --activateUnderlayTraffic=0 --bigSignalingSize=35328 --groundTruthFrequence=1 --pingAsObs=1 --load_path=None --loss_penalty_type=fixed --snapshot_interval=0 --smart_exploration=0 --lambda_train_step=-1 --buffer_soft_limit=0 --lambda_lr=1.0000000000000002e-06 --lamda_training_start_time=0 --d_t_max_time=10 --pingPacketIntervalTime=0.1 --numEpisodes=1 --d_t_send_all_destinations=0 --rcpo_consider_loss=1 --reset_exploration=0 --rcpo_use_loss_pkts=1 --tunnels_max_delays_file_name=examples/5n_overlay_full_mesh_abilene/topology_files/max_observed_values.txt --load_path="examples/5n_overlay_full_mesh_abilene/pre_trained_models/dqn_buffer"'
        
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        print(output)
        self.assertEqual(output, "Hello, world!")

if __name__ == "__main__":
    unittest.main()