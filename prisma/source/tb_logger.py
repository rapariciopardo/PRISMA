
# -*- coding: utf-8 -*-

from tensorboard.plugins.custom_scalar import summary as cs_summary
from tensorboard.plugins.custom_scalar import layout_pb2
import tensorflow as tf
import numpy as np
from time import time

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

def custom_plots():
    
    """define the costum plots for tensorboard
    The user can define the plots he wants to see in tensorboard custom plots, by adding an item to the list of the category
    """
    cs = cs_summary.pb(
            layout_pb2.Layout(
                category=[
                    layout_pb2.Category(
                        title="Main evaluation metrics",
                        chart=[
                            layout_pb2.Chart(
                                title="Avg Delay per arrived pkts",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"avg_delay_over_time"])),
                            layout_pb2.Chart(
                                title="Avg Cost per arrived pkts",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"avg_cost_over_time"])),
                            layout_pb2.Chart(
                                title="Loss Ratio",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"loss_ratio_over_time"])),
                        ]),
                    # layout_pb2.Category(
                    #     title="Global info about the env",
                    #     chart=[
                    #         layout_pb2.Chart(
                    #             title="Average hops over time",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"avg_hops_over_time"])),
                    #         layout_pb2.Chart(
                    #             title="total rewards with and without loss",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"(total_rewards_with_loss_over_time|total_rewards_over_time)"])),
                    #         layout_pb2.Chart(
                    #             title="Buffers occupation",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"nb_buffered_pkts_over_time"])),
                    #         layout_pb2.Chart(
                    #             title="new pkts vs lost pkts vs arrived pkts",
                    #             multiline=layout_pb2.MultilineChartContent(tag=[r"(total_new_rcv_pkts_over_time | total_lost_pkts_over_time | total_arrived_pkts_over_time)"])),
                    #     ]),
                    layout_pb2.Category(
                        title="Training metrics",
                        chart=[
                            layout_pb2.Chart(
                                title="Td error",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"MSE_loss_over_time"])),
                            layout_pb2.Chart(
                                title="exploration value",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"exploaration_value_over_time"])),
                            layout_pb2.Chart(
                                title="replay buffers length",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"replay_buffer_length_over_time"])),
                        ]),
                ]
            )
        )
    return cs

def stats_writer_train(summary_writer_session, summary_writer_nb_arrived_pkts, summary_writer_nb_lost_pkts, summary_writer_nb_new_pkts, Agent):
    """ Write the stats of the session to the logs dir using tensorboard writer during the training
    Args:
        summary_writer_session: main session writer for the reward, loss, delay and nb buffered pkts
        summary_writer_nb_arrived_pkts: writer for nb arrived pkts
        summary_writer_nb_lost_pkts: writer for nb lost pkts
        summary_writer_nb_new_pkts: writer for nb new pkts
        Agent: the agent class object to retrieve static variables and log them
    """
    ## write the global stats
    if Agent.sim_injected_packets > 0:
        loss_ratio = Agent.sim_dropped_packets/Agent.sim_injected_packets
    else:
        loss_ratio = -1
    if Agent.sim_delivered_packets > 0:
        avg_delay = Agent.sim_avg_e2e_delay
        avg_cost = Agent.sim_cost
        avg_hops = Agent.total_hops/Agent.sim_delivered_packets
    else:
        avg_delay = -1
        avg_cost = -1
        avg_hops = -1

    with summary_writer_session.as_default():
        ## total rewards
        tf.summary.scalar('total_e2e_delay_over_iterations', Agent.total_e2e_delay, step=Agent.total_nb_iterations)
        tf.summary.scalar('total_e2e_delay_over_time', Agent.total_e2e_delay, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('total_rewards_with_loss_over_iterations', Agent.total_rewards_with_loss, step=Agent.total_nb_iterations)
        tf.summary.scalar('total_rewards_with_loss_over_time', Agent.total_rewards_with_loss, step=int(Agent.curr_time*1e6))
        ## loss ratio
        tf.summary.scalar('loss_ratio_over_time', loss_ratio, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('loss_ratio_over_iterations', loss_ratio, step=Agent.total_nb_iterations)
        ## total hops and avg hops
        tf.summary.scalar('total_hops_over_iterations', Agent.total_hops, step=Agent.total_nb_iterations)
        tf.summary.scalar('total_hops_over_time', Agent.total_hops, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('avg_hops_over_iterations', avg_hops, step=Agent.total_nb_iterations)
        tf.summary.scalar('avg_hops_over_time', avg_hops, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('ma_avg_hops_over_iterations', np.array(Agent.nb_hops).mean(), step=Agent.total_nb_iterations)
        tf.summary.scalar('ma_avg_hops_over_time', np.array(Agent.nb_hops).mean(), step=int(Agent.curr_time*1e6))
        ## buffers occupation
        tf.summary.scalar('nb_buffered_pkts_over_time', Agent.sim_buffered_packets, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('nb_buffered_pkts_over_iterations', Agent.sim_buffered_packets, step=Agent.total_nb_iterations)
        ## signalling overhead
        tf.summary.scalar('overlay_data_pkts_injected_bytes_time', Agent.sim_bytes_data, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('overlay_big_signalling_bytes', Agent.sim_bytes_big_signaling, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('overlay_small_signalling_bytes', Agent.sim_bytes_small_signaling, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('overlay_ping_signalling_bytes', Agent.sim_bytes_overlay_signaling_back + Agent.sim_bytes_overlay_signaling_forward, step=int(Agent.curr_time*1e6))
        ## avg cost and avg delay
        tf.summary.scalar('avg_cost_over_iterations', avg_cost, step=Agent.total_nb_iterations)
        tf.summary.scalar('avg_cost_over_time', avg_cost, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('avg_delay_over_iterations', avg_delay, step=Agent.total_nb_iterations)
        tf.summary.scalar('avg_delay_over_time', avg_delay, step=int(Agent.curr_time*1e6))
        tf.summary.scalar('ma_delays_over_iterations', np.array(Agent.delays).mean(), step=Agent.total_nb_iterations)
        tf.summary.scalar('ma_delays_over_time', np.array(Agent.delays).mean(), step=int(Agent.curr_time*1e6))
        ## simulation time / real time
        tf.summary.scalar('sim_second_per_real_seconds', (time()-Agent.start_time)/Agent.curr_time, step=int(Agent.curr_time*1e6))

    with summary_writer_nb_arrived_pkts.as_default():
        tf.summary.scalar('pkts_over_iterations', Agent.sim_delivered_packets, step=Agent.total_nb_iterations)
        tf.summary.scalar('pkts_over_time', Agent.sim_delivered_packets, step=int(Agent.curr_time*1e6))

    with summary_writer_nb_lost_pkts.as_default():
        tf.summary.scalar('pkts_over_iterations', Agent.sim_dropped_packets, step=Agent.total_nb_iterations)
        tf.summary.scalar('pkts_over_time', Agent.sim_dropped_packets, step=int(Agent.curr_time*1e6))

    with summary_writer_nb_new_pkts.as_default():
        tf.summary.scalar('pkts_over_iterations', Agent.sim_injected_packets, step=Agent.total_nb_iterations)
        tf.summary.scalar('pkts_over_time', Agent.sim_injected_packets, step=int(Agent.curr_time*1e6))
        
def stats_writer_test(summary_writer_results_path, Agent):
    """ Write the stats of the session to the logs dir using tensorboard writer during test phase
    Args:
        summary_writer_results_path: str. Should be the same as the one used during training
        Agent: the agent class object to retrieve static variables and log them
    """
    model_version = Agent.model_version
    ## create the writer
    summary_writer_results = tf.summary.create_file_writer(logdir=f"{summary_writer_results_path}/{model_version}")
    ## store test stats
    with summary_writer_results.as_default():
        tf.summary.scalar(f'test_global_injected_pkts', Agent.sim_global_injected_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_overlay_injected_pkts', Agent.sim_injected_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_global_lost_pkts', Agent.sim_global_dropped_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_overlay_lost_pkts', Agent.sim_dropped_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_global_arrived_pkts', Agent.sim_global_delivered_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_overlay_arrived_pkts', Agent.sim_delivered_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_global_e2e_delay', Agent.sim_avg_e2e_delay, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_overlay_e2e_delay', Agent.sim_global_avg_e2e_delay, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_global_loss_rate', Agent.sim_global_dropped_packets/Agent.sim_global_injected_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_overlay_loss_rate', Agent.sim_dropped_packets/Agent.sim_injected_packets, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_global_cost', Agent.sim_global_cost, step=int(Agent.load_factor*100))
        tf.summary.scalar(f'test_overlay_cost', Agent.sim_cost, step=int(Agent.load_factor*100))
        # tf.summary.scalar('test_global_e2e_delay', Agent.sim_avg_e2e_delay, step=int(Agent.load_factor*100))
        # tf.summary.scalar('test_global_loss_rate', Agent.sim_global_dropped_packets/Agent.sim_global_injected_packets, step=int(Agent.load_factor*100))     
