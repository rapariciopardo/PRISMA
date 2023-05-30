
__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

def run_ns3(params, configure=True):
    """
    Run the ns3 simulator
    Args: 
        params(dict): parameter dict
        configure(bool): if True, run ns3 configure
    Returns:
        proc: process id of the ns3 simulator
    """ 
    ## import libraries
    import os
    import shlex
    import subprocess
    
    ## check if ns3-gym is in the folder
    if "waf" not in os.listdir(params["ns3_sim_path"]):
        raise Exception(f'Unable to locate ns3-gym in the folder : {params["ns3_sim_path"]}')
        
    ## store current folder path
    current_folder_path = os.getcwd()

    ## Copy prisma into ns-3 folder
    os.system(f'rsync -r ./ns3/* {params["ns3_sim_path"].rstrip("/")}/scratch/prisma')
    os.system(f'rsync -r ./ns3_model/ipv4-interface.cc {params["ns3_sim_path"].rstrip("/")}/src/internet/model')

    ## go to ns3 dir
    os.chdir(params["ns3_sim_path"])
    
    ## run ns3 configure
    #configure_command = './waf -d optimized configure'
    if configure:
        os.system('./waf configure')

    ## run NS3 simulator
    ns3_params_format = ('prisma --simSeed={} --openGymPort={} --simTime={} --AvgPacketSize={} '
                        '--LinkDelay={} --LinkRate={} --MaxBufferLength={} --load_factor={} '
                        '--adj_mat_file_name={} --overlay_mat_file_name={} --node_coordinates_file_name={} '
                        '--node_intensity_file_name={} --signaling={} --AgentType={} --signalingType={} '
                        '--syncStep={} --lossPenalty={} --activateOverlaySignaling={} --nPacketsOverlaySignaling={} '
                        '--train={} --movingAverageObsSize={} --activateUnderlayTraffic={} --opt_rejected_file_name={} '
                        '--map_overlay_file_name={} --pingAsObs={} --logs_folder={} --groundTruthFrequence={} --bigSignalingSize={}'.format( params["seed"],
                                             params["basePort"],
                                             str(params["simTime"]),
                                             params["packet_size"],
                                             params["link_delay"],
                                             str(params["link_cap"]) + "bps",
                                             str(params["max_out_buffer_size"]) + "B",
                                             params["load_factor"],
                                             params["physical_adjacency_matrix_path"],
                                             params["overlay_adjacency_matrix_path"],
                                             params["node_coordinates_path"],
                                             params["traffic_matrix_path"],
                                             bool(params["signalingSim"]),
                                             params["agent_type"],
                                             params["signaling_type"],
                                             params["sync_step"],
                                             params["loss_penalty"],
                                             bool(params["activateOverlay"]),
                                             params["nPacketsOverlay"],
                                             bool(params["train"]),
                                             params["movingAverageObsSize"],
                                             bool(params["activateUnderlayTraffic"]),
                                             params["opt_rejected_path"],
                                             params["map_overlay_path"],
                                             bool(params["pingAsObs"]),
                                             params["logs_folder"],
                                             params["groundTruthFrequence"],
                                             params["bigSignalingSize"]
                                             ))
    run_ns3_command = shlex.split(f'./waf --run "{ns3_params_format}"')
    proc = subprocess.Popen(run_ns3_command)
    print(f"Running ns3 simulator with process id: {proc.pid}")
    os.chdir(current_folder_path)
    return proc.pid
