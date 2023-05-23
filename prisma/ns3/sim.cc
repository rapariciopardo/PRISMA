/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo and Lucile Sassatelli
 *
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Redha A. Alliche, <alliche@i3s.unice.fr,>
 * Author: Tiago Da Silva Barros    <tiago.da-silva-barros@inria.fr>
 * Author: Ramon Aparicio-Pardo       <raparicio@i3s.unice.fr,>
 * Author: Lucile Sassatelli       <sassatelli@i3s.unice.fr,>
 *
 * Université Côte d’Azur, CNRS, I3S, Inria Sophia Antipolis, France
 *
 * Work supported in part by he  support  of  the  French  Agence  Nationale  dela Recherche (ANR), 
 * under grant ANR-19-CE-25-0001-01 (ARTIC project).
 * This  work  was  performed  using  HPC  resources  from  GENCI-IDRIS  (Grant2021-AD011012577).
 * 
 * This work is partially based on Copyright (c) 2010 Egemen K. Cetinkaya, Justin P. Rohrer, and Amit Dandekar
 * available on https://www.nsnam.org/doxygen/matrix-topology_8cc_source.html
 *
 * This program reads an upper triangular adjacency matrix (e.g. adjacency_matrix.txt),
 * node coordinates file (e.g. node_coordinates.txt) an a traffic rate matrix (e.g. nodes_intensity_normalized.txt). 
 * The program also set-ups a wired network topology with P2P links according to the adjacency matrix with
 * nx(n-1) CBR traffic flows, in which n is the number of nodes in the adjacency matrix. Then the programs makes a callback funtion for each time
 * a packet arrives a node.
 */

// ---------- Header Includes -------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "point-to-point-helper.h"
#include "point-to-point-net-device.h"
#include "ns3/traffic-control-helper.h" 
#include "ns3/applications-module.h"
#include "ns3/virtual-net-device.h"


#include "poisson-app-helper.h"
#include "big-signaling-app-helper.h"
#include "big-signaling-application.h"
#include "tcp-application.h"
#include "ns3/global-route-manager.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/assert.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "packet-routing-gym.h"
#include "data-packet-manager.h"
#include "ipv4-ospf-routing.h"
#include "conf-loader.h"


using namespace std;
using namespace ns3;




//NS_LOG_COMPONENT_DEFINE ("OpenGym");

// ---------- Prototypes ------------------------------------------------------

vector<vector<bool> > readNxNMatrix (std::string adj_mat_file_name);
vector<vector<double> > readCordinatesFile (std::string node_coordinates_file_name);
vector<vector<std::string>> readIntensityFile(std::string intensity_file_name);
vector<vector<double> > readOptRejectedFile (std::string opt_rejected_file_name);
vector<int> readOverlayMatrix(std::string overlay_file_name);
void printCoordinateArray (const char* description, vector<vector<double> > coord_array);
void printMatrix (const char* description, vector<vector<bool> > array);
void ScheduleNextTrainStep(Ptr<PacketRoutingEnv> openGym);
void ScheduleHelloMessages(Ipv4OSPFRouting* ospf);
void RestoreLinkRate(NetDeviceContainer *ptp, u_int32_t idx1, u_int32_t idx2) {
  NS_LOG_UNCOND("tempo: "<<Simulator::Now());
  NS_LOG_UNCOND(idx1<<"     "<<idx2<< "    aqui    ");
  StaticCast<PointToPointNetDevice>(ptp->Get(idx1))->NotifyLink(true);
  StaticCast<PointToPointNetDevice>(ptp->Get(idx2))->NotifyLink(true);
    
}
void ModifyLinkRate(NetDeviceContainer *ptp, u_int32_t idx1, u_int32_t idx2, double duration) {
  NS_LOG_UNCOND(idx1<<"     "<<idx2<< "    aqui    ");
  StaticCast<PointToPointNetDevice>(ptp->Get(idx1))->NotifyLink(false);
  StaticCast<PointToPointNetDevice>(ptp->Get(idx2))->NotifyLink(false);
  Simulator::Schedule(Seconds(duration), &RestoreLinkRate, ptp, idx1, idx2);
}



int counter_send[100]= {0};
void countPackets(int n_nodes, NetDeviceContainer* nd, std::string path, Ptr<const Packet> packet, const Address &src, const Address &dest){

  Ptr<Packet> p = packet->Copy();
  NS_LOG_UNCOND(p->ToString());
}





NS_LOG_COMPONENT_DEFINE ("GenericTopologyCreation");

int main (int argc, char *argv[])
{

  // ---------- Simulation Variables ------------------------------------------

  // Change the variables and file names only in this block!
  // Parameters of the environment
  uint32_t simSeed = 1;
  uint32_t openGymPort = 6555;
  uint32_t testArg = 0;
  double simTime = 60; //seconds
  double envStepTime = 0.05; //seconds, ns3gym env step time interval
  double linkFailureTime = 25; //seconds
  int nblinksFailed = 3;
  double linkFailureDuration = 3;
  int nbNodesUpdated = 1;
  double updateTrafficRateTime = 10.0;
  bool perturbations = false;
  bool activateSignaling = false;
  std::string agentType("DQN-buffer");
  std::string signalingType("NN");
  double syncStep = 1.0;
  double bigSignalingSize = 35328;
  bool eventBasedEnv = true;
  double load_factor = 0.01; // scaling applied to the traffic matrix
  std::string MaxBufferLength ("30p");
  std::string LinkRate ("500Kbps");
  std::string BadLinkRate ("500Kbps");
  std::string LinkDelay ("2ms");
  uint32_t AvgPacketSize = 512 ; //—> If you want to change the by-default 512 packet size

  std::string adj_mat_file_name ("scratch/prisma/examples/abilene/adjacency_matrix.txt");
  std::string overlay_mat_file_name ("scratch/prisma/examples/abilene/overlay_matrix.txt");
  std::string node_coordinates_file_name ("scratch/prisma/examples/abilene/node_coordinates.txt");
  std::string node_intensity_file_name("scratch/prisma/examples/abilene/node_intensity.txt");
  std::string opt_rejected_file_name("scratch/prisma/test.txt");
  std::string map_overlay_file_name("scratch/prisma/test2.txt");
  std::string logs_folder("../prisma/abilene/examples/4n");

  float groundTruthFrequence = -1;


  bool activateOverlaySignaling = true;
  uint32_t nPacketsOverlaySignaling = 2;

  double lossPenalty = 0.0;

  bool train = false;

  bool pingAsObs = true;

  uint32_t movingAverageObsSize = 5;

  bool activateUnderlayTraffic = true;
  
  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("eventBasedEnv", "Whether steps should be event or time based. Default: true", eventBasedEnv);
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 30s", simTime);
  cmd.AddValue ("adj_mat_file_name", "Adjacency matrix file path. Default: scratch/prisma/adjacency_matrix.txt", adj_mat_file_name);
  cmd.AddValue ("overlay_mat_file_name", "Adjacency matrix file path. Default: scratch/prisma/overlay_matrix.txt", overlay_mat_file_name); 
  cmd.AddValue ("node_coordinates_file_name", "Node coordinates file path. Default: scratch/prisma/node_coordinates.txt", node_coordinates_file_name);
  cmd.AddValue ("node_intensity_file_name", "Node intensity (traffic matrix) file path. Default: scratch/prisma/node_intensity.txt", node_intensity_file_name);
  cmd.AddValue ("AvgPacketSize", "Packet size. Default: 512", AvgPacketSize);
  cmd.AddValue ("LinkDelay", "Network links delay. Default: 2ms", LinkDelay);
  cmd.AddValue ("LinkRate", "Network links capacity in bits per seconds. Default: 500Kbps", LinkRate);
  cmd.AddValue ("MaxBufferLength", "Output buffers max size. Default: 30p", MaxBufferLength);
  cmd.AddValue ("load_factor", "scale of the traffic matrix. Default: 1.0", load_factor);
  cmd.AddValue ("stepTime", "Gym Env step time in seconds. Default: 0.1s", envStepTime);
  cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.AddValue ("linkFailure", "link Failure time. Default: 30", linkFailureTime);
  cmd.AddValue ("nbLinksFailed", "Number of links failing. Default: 3", nblinksFailed);
  cmd.AddValue ("linkFailureDuration", "Duration of the link Failure. Default: 10 s", linkFailureDuration);
  cmd.AddValue ("nbNodesUpdated", "Number of nodes to be updated (Average Traffic Rate). Default: 1", nbNodesUpdated);
  cmd.AddValue ("updateTrafficRateTime", "Frequency to update Traffic rate. Default: 10.0s", updateTrafficRateTime);
  cmd.AddValue ("perturbations", "Adding perturbations for the network. Default: false", perturbations);
  cmd.AddValue ("signaling", "Adding signaling to the NS3 simulation", activateSignaling);
  cmd.AddValue ("AgentType", "Agent Type", agentType);
  cmd.AddValue ("signalingType", "Signaling Type", signalingType);
  cmd.AddValue ("syncStep", "synchronization Step (in seconds)", syncStep);
  cmd.AddValue ("activateOverlaySignaling", "activate Overlay Signaling", activateOverlaySignaling);
  cmd.AddValue ("nPacketsOverlaySignaling", "nb of packets for triggering overlay signaling", nPacketsOverlaySignaling);
  cmd.AddValue ("lossPenalty", "Packet Loss Penalty", lossPenalty);
  cmd.AddValue ("train", "train", train);
  cmd.AddValue ("movingAverageObsSize", "size of MA for collecting the Obs", movingAverageObsSize);
  cmd.AddValue ("activateUnderlayTraffic", "Set if activate underlay traffic", activateUnderlayTraffic);
  cmd.AddValue ("opt_rejected_file_name", "Rejected paths in Optimal Algorithm file name", opt_rejected_file_name);
  cmd.AddValue ("map_overlay_file_name", "Map overlay file name", map_overlay_file_name);
  cmd.AddValue ("pingAsObs", "ping as observation variable", pingAsObs);
  cmd.AddValue ("logs_folder", "Logs folder", logs_folder);
  cmd.AddValue ("groundTruthFrequence", "ground truth freq", groundTruthFrequence);
  cmd.AddValue ("bigSignalingSize", "total size of the weights of the NN in bytes", bigSignalingSize);

  cmd.Parse (argc, argv);
    
  NS_LOG_UNCOND("Ns3Env parameters:");
  NS_LOG_UNCOND("--simulationTime: " << simTime);
  NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND("--seed: " << simSeed);
  NS_LOG_UNCOND("--testArg: " << testArg);
  NS_LOG_UNCOND("--adj_mat_file_name: " << adj_mat_file_name);
  NS_LOG_UNCOND("--node_coordinates_file_name: " << node_coordinates_file_name);
  NS_LOG_UNCOND("--node_intensity_file_name: " << node_intensity_file_name);
  NS_LOG_UNCOND("--LinkDelay: " << LinkDelay);
  NS_LOG_UNCOND("--MaxBufferLength: " << MaxBufferLength);
  NS_LOG_UNCOND("--load_factor: " << load_factor);
  NS_LOG_UNCOND("--linkFailureTime: "<<linkFailureTime);
  NS_LOG_UNCOND("--Signaling: "<<activateSignaling);
  NS_LOG_UNCOND("--agentType: "<< agentType);
  NS_LOG_UNCOND("--SignalingType: "<<signalingType);
  NS_LOG_UNCOND("--lossPenalty: "<<lossPenalty);
  NS_LOG_UNCOND("--activateUnderlayTraffic: "<<activateUnderlayTraffic);
  NS_LOG_UNCOND("--RejectedTraffPath: "<<opt_rejected_file_name);
  NS_LOG_UNCOND("--pingAsObs: "<<pingAsObs);
  NS_LOG_UNCOND("--movingAverageObsSize: "<<movingAverageObsSize);
  NS_LOG_UNCOND("--nPacketsOverlaySignaling: "<<nPacketsOverlaySignaling);

  
  ComputeStats compStats;
  compStats.setLossPenalty(lossPenalty);
  //Parameters of the scenario
  double SinkStartTime  = 0.0001;
  double SinkStopTime   = simTime; // - 0.1;//59.90001;
  double AppStartTime   = 0.0001;
  double AppStopTime    = simTime; // - 0.2;

  NS_LOG_UNCOND("START/STOP TIME"<<AppStartTime<<"   "<<AppStopTime);

  AvgPacketSize = AvgPacketSize; // remove the header length 8 20 18

  
    
  RngSeedManager::SetSeed (simSeed);
  RngSeedManager::SetRun (simSeed);
  
  LogComponentEnable ("GenericTopologyCreation", LOG_NONE );


  // srand ( (unsigned)time ( NULL ) );   // generate different seed each time

  srand (simSeed);   // fix the seed for the random number generator

  std::string tr_name ("n-node-ppp.tr");
  std::string pcap_name ("n-node-ppp");
  std::string flow_name ("n-node-ppp.xml");
  std::string anim_name ("n-node-ppp.anim.xml");



  // ---------- End of Simulation Variables ----------------------------------

  // ---------- Read Adjacency Matrix ----------------------------------------

  vector<vector<bool> > Adj_Matrix;
  Adj_Matrix = readNxNMatrix (adj_mat_file_name);

  vector<vector<bool> > OverlayAdj_Matrix;
  OverlayAdj_Matrix = readNxNMatrix (overlay_mat_file_name);
  NS_LOG_UNCOND("OverlayAdj_Matrix size: "<<OverlayAdj_Matrix.size() << " name: "<<overlay_mat_file_name );

  vector<int> underlay_to_overlay_map;
  underlay_to_overlay_map = readOverlayMatrix(map_overlay_file_name);

  vector<vector<double>> OptRejected;
  OptRejected = readOptRejectedFile(opt_rejected_file_name);

  //int n_packetsDropped = 0;
  //int n_packetsInjected = 0;
  //int n_packetsDelivered = 0;



  // Optionally display 2-dimensional adjacency matrix (Adj_Matrix) array
  // printMatrix (adj_mat_file_name.c_str (),Adj_Matrix);

  // ---------- End of Read Adjacency Matrix ---------------------------------

  // ---------- Read Node Coordinates File -----------------------------------



  vector<vector<double> > coord_array;
  coord_array = readCordinatesFile (node_coordinates_file_name);

  vector<vector<std::string>> Traff_Matrix;
  Traff_Matrix = readIntensityFile (node_intensity_file_name);

  NS_LOG_UNCOND(node_intensity_file_name);
  int n_nodes = coord_array.size () ; //coord_array.size ();
  int overlay_n_nodes = OverlayAdj_Matrix.size() ; //coord_array.size ();
  int matrixDimension = Adj_Matrix.size ();

  if (matrixDimension != n_nodes)
    {
      NS_FATAL_ERROR ("The number of lines in coordinate file is: " << n_nodes << " not equal to the number of nodes in adjacency matrix size " << matrixDimension);
    }

  // ---------- End of Read Node Coordinates File ----------------------------
  // ---------- Setting overlay network---------------------------------------


  
  // ---------- End of Setting overlay network---------------------------------------

  // ---------- Network Setup ------------------------------------------------

  //Create Node Containers
  NS_LOG_INFO ("Create Nodes.");

  NodeContainer nodes_switch;
  nodes_switch.Create(n_nodes);
  
  NodeContainer nodes_traffic;   // Declare nodes objects
  nodes_traffic.Create (n_nodes);

  //Create Net Devices
  NS_LOG_INFO ("Create P2P Link Attributes.");

  NetDeviceContainer traffic_nd;
  NetDeviceContainer switch_nd;

  // Computing node degrees for phyiscal network
  std::vector<NetDeviceContainer> dev_links;
  int nodes_degree[n_nodes] ={0};
  for (size_t i = 0; i < Adj_Matrix.size (); i++)
      {
        for (size_t j = 0; j < Adj_Matrix[i].size (); j++)
          {
            if (Adj_Matrix[i][j] == 1)
              {
                nodes_degree[i] += 1;
              } 
          }
      }

  // Computing node degrees for overlay network
  int overlay_nodes_degree[overlay_n_nodes] ={0};
  for (size_t i = 0; i < OverlayAdj_Matrix.size (); i++)
      {
        for (size_t j = 0; j < OverlayAdj_Matrix[i].size (); j++)
          {
            if (OverlayAdj_Matrix[i][j] == 1)
              {
                overlay_nodes_degree[i] += 1;
              } 
          }
      }


  //Creating the IP Stack Helpers
  InternetStackHelper internet;
  internet.Install(nodes_traffic);
  internet.Install(nodes_switch);

  //Parameters of signaling
  double smallSignalingSize[overlay_n_nodes] = {0.0};
  if(agentType=="sp" || agentType=="opt" || signalingType=="ideal"){
    activateSignaling=false;
  }
  bool opt = false;
  if(agentType=="opt") opt=true;
  if(signalingType=="NN"){
    NS_LOG_UNCOND("SMALL SIGNALING");
    for(int i=0;i<overlay_n_nodes;i++){
      smallSignalingSize[i] = 8 + (8 * (overlay_nodes_degree[i]+1));
    }
  } else if(signalingType=="target"){
    for(int i=0;i<overlay_n_nodes;i++){
      smallSignalingSize[i] = 24;
    }
  }
    else if(signalingType=="digital_twin"){
      for(int i=0;i<overlay_n_nodes;i++){
        smallSignalingSize[i] = 8 + (8 * (2* overlay_nodes_degree[i] + 1));
      }
  }  

  //Creating the links
  NS_LOG_UNCOND("Creating link between traffic generator nodes and switch nodes in physical network");
  
  for(int i=0;i<n_nodes;i++)
  {
    PointToPointHelper p2p;
    DataRate data_rate(LinkRate);

    p2p.SetDeviceAttribute ("DataRate", DataRateValue (1000000*data_rate.GetBitRate()*nodes_degree[i]));
    p2p.SetChannelAttribute ("Delay", StringValue ("0ms"));
    p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("1000p"));    
    NetDeviceContainer n_devs = p2p.Install (NodeContainer (nodes_traffic.Get(i), nodes_switch.Get(i)));
    dev_links.push_back(n_devs);
    traffic_nd.Add(n_devs.Get(0));
    switch_nd.Add(n_devs.Get(1));
  }
  vector<tuple<int, int>> link_devs;
  Ptr<NetDevice> nds_switch [n_nodes][n_nodes];
  NS_LOG_UNCOND("Creating link between switch nodes physical network");
  for (size_t i = 0; i < Adj_Matrix.size(); i++)
      {
        for (size_t j = i; j < Adj_Matrix[i].size(); j++)
          {
            if (Adj_Matrix[i][j] == 1)
              {
                PointToPointHelper p2p;

                p2p.SetDeviceAttribute ("DataRate", DataRateValue (LinkRate));
                p2p.SetChannelAttribute ("Delay", StringValue (LinkDelay));
                p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue (MaxBufferLength));    
                NetDeviceContainer n_devs = p2p.Install(NodeContainer(nodes_switch.Get(i), nodes_switch.Get(j)));
                switch_nd.Add(n_devs.Get(0));
                switch_nd.Add(n_devs.Get(1));
                nds_switch[i][j] = n_devs.Get(0);
                nds_switch[j][i] = n_devs.Get(1);
                link_devs.push_back(make_tuple(switch_nd.GetN()-1, switch_nd.GetN()-2));
              }
          }
      }
  //for(size_t i=0;i<link_devs.size();i++){
  //  NS_LOG_UNCOND(get<0>(link_devs[i]) << "     " << get<1>(link_devs[i])<<"    "<<switch_nd.Get(get<0>(link_devs[i]))->GetNode()->GetId()<<"     "<<switch_nd.Get(get<1>(link_devs[i]))->GetNode()->GetId());
  //  NS_LOG_UNCOND(switch_nd.Get(get<0>(link_devs[i]))->GetIfIndex());
  //}
  


  //Adding Ipv4 Address to the Net Devices for traffic generator nodes
  Ipv4AddressHelper address;
  address.SetBase ("10.2.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces_traffic = address.Assign (traffic_nd);
  NS_LOG_UNCOND("hhh" << traffic_nd.GetN());
  // Adding Ipv4 Address to the Net Devices for switch nodes
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces_switch = address.Assign (switch_nd);
  TrafficControlHelper tch;
  tch.Uninstall (switch_nd);
  
  // Create the overlay network map
  //Specifies the UnderlayIndex of overlay Nodes
  int overlay_to_underlay_map[overlay_n_nodes];

  //Specifies the underlay indexes of neighbors of overlay nodes (by underlayIndex)
  vector<int> overlayNeighbors[overlay_n_nodes];

  //Check by underlayIndex if a node is Overlay
  bool overlayNodesChecker[n_nodes] = {0};

  //Check for a node and store the mapping between overlay to underlay nodes (overlay_to_underlay_map[overlayIndex] = underlayIndex])
  for(int i=0;i<n_nodes;i++){
    if(underlay_to_overlay_map[i]>=0){
      overlayNodesChecker[i] = 1;
      overlay_to_underlay_map[underlay_to_overlay_map[i]]=i;
    }
  }
  // Construct the overlay nodes neighbors 
  for(int i=0;i<overlay_n_nodes;i++){
    for(int j=i;j<overlay_n_nodes;j++){
      if(OverlayAdj_Matrix[i][j]==1){
        overlayNeighbors[i].push_back(overlay_to_underlay_map[j]);
        overlayNeighbors[j].push_back(overlay_to_underlay_map[i]);
      }
    }
  }

  // print overlay nodes and neighbors
  for(int i=0;i<overlay_n_nodes;i++){
      NS_LOG_UNCOND(" " <<"Overlay Node: "<<i<<"  Neighbors: ");
      for(int j=0;j<int(overlayNeighbors[i].size());j++){
        NS_LOG_UNCOND(" " <<overlayNeighbors[i][j]<<" ");
      }
      NS_LOG_UNCOND(" ");
  }
  // Computing the starting ip address for a node
  int nodes_starting_address[n_nodes]={0};
  for (int i=1;i<n_nodes;i++){
    nodes_starting_address[i] = nodes_starting_address[i-1] + nodes_degree[i];
    NS_LOG_UNCOND(" printing overlay starting adress for node  " << i << " " <<nodes_starting_address[i]);
  }

  //Create The Overlay Mask Traffic rate
  float ActivateTrafficRate[n_nodes][n_nodes];
  float OverlayMaskTrafficRate[n_nodes][n_nodes];

  for(int i=0;i<n_nodes;i++){
    for(int j=0;j<n_nodes;j++){

      if(overlayNodesChecker[i] && overlayNodesChecker[j] && i!=j){
        OverlayMaskTrafficRate[i][j] = 1.0;
        ActivateTrafficRate[i][j] = 1.0;
      } else{
        OverlayMaskTrafficRate[i][j] =0.0;
        ActivateTrafficRate[i][j] = 0.0;
      }
      if(activateUnderlayTraffic){
        ActivateTrafficRate[i][j] = 1.0;
      }
      std::cout<<OverlayMaskTrafficRate[i][j]<<" ";

    }
    std::cout<<std::endl;
  }

  // OpenGym Env
  NS_LOG_INFO ("Setting up OpemGym Envs for each node.");
  
  std::vector<Ptr<OpenGymInterface> > myOpenGymInterfaces;
  std::vector<Ptr<PacketRoutingEnv> > packetRoutingEnvs;
  
  uint64_t linkRateValue= DataRate(LinkRate).GetBitRate();
  for (int i = 0; i < n_nodes; i++)
  {
    NS_LOG_UNCOND("Physical Node: "<<i);
    NS_LOG_UNCOND("nb Neighbors: " << nodes_degree[i]);
  }
  for (int i = 0; i < overlay_n_nodes; i++)
  {
    NS_LOG_UNCOND("Node: "<<overlay_to_underlay_map[i]<<"   Port: "<<openGymPort + i);
    NS_LOG_UNCOND("Neighbors: ");
    for(int neighbor : overlayNeighbors[i]){
      NS_LOG_UNCOND(neighbor);
    }

    
    Ptr<Node> n = nodes_switch.Get (overlay_to_underlay_map[i]); // ref node
    //nodeOpenGymPort = openGymPort + i;
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort + i);
    Ptr<PacketRoutingEnv> packetRoutingEnv;
    packetRoutingEnv = CreateObject<PacketRoutingEnv> (n, n_nodes, linkRateValue, activateSignaling, smallSignalingSize[i], overlayNeighbors[i], nodes_starting_address); // event-driven step
    packetRoutingEnv->setTrainConfig(train);
    packetRoutingEnv->m_nodes = nodes_switch;
    packetRoutingEnv->mapOverlayNodes(underlay_to_overlay_map);
    //packetRoutingEnv->setLogsFolder(logs_folder);
    //packetRoutingEnv->setOverlayConfig(overlayNeighbors[overlay_to_underlay_map[i]], activateOverlaySignaling, nPacketsOverlaySignaling, movingAverageObsSize, underlay_to_overlay_map);
    packetRoutingEnv->SetOpenGymInterface(openGymInterface);
    packetRoutingEnv->initialize();
    //packetRoutingEnv->m_node_container = &nodes_switch;
    //packetRoutingEnv->setPingTimeout(16260, 500000, 1);
    //packetRoutingEnv->setLossPenalty(lossPenalty);
    packetRoutingEnv->setNetDevicesContainer(&switch_nd);
    packetRoutingEnv->configDataPacketManager(!pingAsObs, nPacketsOverlaySignaling);
    packetRoutingEnv->configPingBackPacketManager(movingAverageObsSize);
    //packetRoutingEnv->setPingAsObs(pingAsObs);
    //if(i==0 && groundTruthFrequence>0){
    //  packetRoutingEnv->setGroundTruthFrequence(groundTruthFrequence);
    //}
    for(size_t j = 1;j<nodes_switch.Get(overlay_to_underlay_map[i])->GetNDevices();j++){
      NS_LOG_UNCOND("Device: "<<overlay_to_underlay_map[i] << "   Port: "<<nodes_switch.Get(overlay_to_underlay_map[i])->GetDevice(j)->GetIfIndex());
      Ptr<NetDevice> dev_switch =DynamicCast<NetDevice> (nodes_switch.Get(overlay_to_underlay_map[i])->GetDevice(j)); 
      dev_switch->TraceConnectWithoutContext("MacRx", MakeBoundCallback(&PacketRoutingEnv::NotifyPktRcv, packetRoutingEnv, dev_switch, &traffic_nd));
    }
    myOpenGymInterfaces.push_back (openGymInterface);
    packetRoutingEnvs.push_back (packetRoutingEnv);
  }  

  // ---------- Create n*(n-1) CBR Flows -------------------------------------

  
  NS_LOG_INFO ("");
  //Not used anymore
  float sum_traffic_rate_mat = 0.0;
  float sum_masked_traffic_rate_mat = 0.0;
  int count_traffic_rate_mat = 0;
  int count_masked_traffic_rate_mat = 0;

  for(int i=0;i<n_nodes;i++){
    for(int j = 0;j<n_nodes;j++){
      if(i!=j){
        sum_traffic_rate_mat += ceil(DataRate(Traff_Matrix[i][j]).GetBitRate());
        count_traffic_rate_mat++;
        if(ActivateTrafficRate[i][j]==1.0){
          //NS_LOG_UNCOND(ceil(DataRate(Traff_Matrix[i][j]).GetBitRate()));
          sum_masked_traffic_rate_mat += ceil(DataRate(Traff_Matrix[i][j]).GetBitRate());
          count_masked_traffic_rate_mat++;
        } 
      }
    }
  }
  //sum_traffic_rate_mat /= n_nodes;
  //if(activateUnderlayTraffic) sum_masked_traffic_rate_mat /= n_nodes;
  //else sum_masked_traffic_rate_mat /= overlay_to_underlay_map.size();
  //float factor_overlay = sum_traffic_rate_mat / sum_masked_traffic_rate_mat;
  ////------------------------------------------------------------------------------------
  float factor_overlay = 1.0;
  //NS_LOG_UNCOND("FACTOR OVERLAY "<<sum_traffic_rate_mat<<"    "<<sum_masked_traffic_rate_mat<<"    "<<factor_overlay);

  NS_LOG_UNCOND("Setup CBR Traffic Sources.");
  
  
  uint16_t sinkPortUDP = 23;
  
  //int interface = 0;
  for (int i = 0; i < n_nodes; i++)
    {
      //interface = 0;
      for (int j = 0; j < n_nodes; j++)
        {
          if (i != j && ActivateTrafficRate[i][j]>0 && DataRate(Traff_Matrix[i][j]).GetBitRate()>0)
            {
              // We needed to generate a random number (rn) to be used to eliminate
              // the artificial congestion caused by sending the packets at the
              // same time. This rn is added to AppStartTime to have the sources
              // start at different time, however they will still send at the same rate.
              Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
              x->SetAttribute ("Min", DoubleValue (0));
              x->SetAttribute ("Max", DoubleValue (1));
              Address sinkAddress;
              string string_ip_dest;
              if(OverlayMaskTrafficRate[i][j]==1.0) string_ip_dest= "10.1.1."+std::to_string(i+1);
              else string_ip_dest= "10.2.2."+std::to_string(j+1);
              Ipv4Address ip_dest(string_ip_dest.c_str());
              sinkAddress = InetSocketAddress (ip_dest, sinkPortUDP);
              if(true){ //((i==10 && j==5) || (i==5 && j==0)){
                double rn = x->GetValue ();
                PoissonAppHelper poisson  ("ns3::UdpSocketFactory",sinkAddress);
                //NS_LOG_UNCOND(i<<"  "<<j<<"   "<<DataRate(ceil(DataRate(Traff_Matrix[i][j]).GetBitRate()*load_factor*factor_overlay)).GetBitRate());
                poisson.SetAverageRate (DataRate(ceil(DataRate(Traff_Matrix[i][j]).GetBitRate()*load_factor*factor_overlay)), AvgPacketSize);
                poisson.SetTrafficValableProbability(OverlayMaskTrafficRate[i][j]);
                //NS_LOG_UNCOND(opt<<"   "<<OptRejected[i][j]);
                poisson.SetRejectedProbability(opt,OptRejected[i][j]);
                poisson.SetUpdatable(false, updateTrafficRateTime);
                poisson.SetDestination(uint32_t (j+1), uint32_t (i+1));
                ApplicationContainer apps = poisson.Install (nodes_traffic.Get (i));
                apps.Start (Seconds (AppStartTime + rn));
                apps.Stop (Seconds (AppStopTime));
              }
              
              if(train && activateSignaling && signalingType=="NN" && overlayNodesChecker[i] && overlayNodesChecker[j]){
                if(OverlayAdj_Matrix[underlay_to_overlay_map[i]][underlay_to_overlay_map[j]]==1 ){
                  string string_ip_bigSignaling= "10.2.2."+std::to_string(j+1);
                  Ipv4Address ip_big_signaling(string_ip_bigSignaling.c_str());
                  sinkAddress = InetSocketAddress (ip_big_signaling, sinkPortUDP);

                  BigSignalingAppHelper sign ("ns3::UdpSocketFactory",sinkAddress); 
                  sign.SetAverageStep (syncStep, bigSignalingSize); 
                  sign.SetSourceDest(i+1, j+1, underlay_to_overlay_map[i]+1); 
                  ApplicationContainer apps = sign.Install (nodes_traffic.Get (i));  // traffic sources are installed on all nodes 
                  apps.Start (Seconds (AppStartTime )); 
                  apps.Stop (Seconds (AppStopTime)); 
                }      
              }
                    
              
            }
        }
    }
  //  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::BigSignalingGeneratorApplication/TxWithAddresses",MakeBoundCallback(&countPackets, n_nodes, &traffic_nd));

 
  
  


  NS_LOG_INFO ("Setup Packet Sinks.");

  //for (int i=0;i<n_nodes;i++){
  //  Address anyAddress;
  //  anyAddress = InetSocketAddress (Ipv4Address::GetAny (), sinkPort);
  //  PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", anyAddress);
  //  ApplicationContainer sinkApps = packetSinkHelper.Install (nodes_switch.Get (i));
  //  sinkApps.Start (Seconds (SinkStartTime));
  //  sinkApps.Stop (Seconds (SinkStopTime));
  //
  //}

  for (int i=0;i<n_nodes;i++){
    Address anyAddress;
    anyAddress = InetSocketAddress (Ipv4Address::GetAny (), sinkPortUDP);
    PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", anyAddress);
    ApplicationContainer sinkApps = packetSinkHelper.Install (nodes_traffic.Get (i));
    sinkApps.Start (Seconds (SinkStartTime));
    sinkApps.Stop (Seconds (SinkStopTime));

  }

  NS_LOG_UNCOND ("Initialize Global Routing.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  
  

  // ---------- End of Create n*(n-1) CBR Flows ------------------------------


  // ---------- Simulation Monitoring ----------------------------------------
  NS_LOG_UNCOND("Configuration of time based alert");
  for(int i=0;i<int(packetRoutingEnvs.size());i++)
  {
    Simulator::Schedule (Seconds(0.0), &ScheduleNextTrainStep, packetRoutingEnvs[i]);
    NS_LOG_UNCOND(i);
  }
  NS_LOG_INFO ("Configure Tracing.");
  
  
  NS_LOG_INFO ("Run Simulation.");
  NS_LOG_UNCOND ("Simulation start");
  ns3::PacketMetadata::Enable();
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  // flowmon->SerializeToXmlFile (flow_name.c_str(), true, true);
  NS_LOG_UNCOND ("Simulation stop");
  PointToPointNetDevice::PrintDelayInfo(agentType, load_factor);
  NS_LOG_UNCOND("Sent Packets: "<< counter_send);
  for (size_t i = 0; i < myOpenGymInterfaces.size(); i++)
  {
    NS_LOG_UNCOND("flags" << i);
    packetRoutingEnvs[i]->is_trainStep_flag = 1;
    //packetRoutingEnvs[i]->simulationEnd(activateUnderlayTraffic, load_factor);
    myOpenGymInterfaces[i]->NotifySimulationEnd();
  }
  Simulator::Destroy ();

  // ---------- End of Simulation Monitoring ---------------------------------

  return 0;

}

// ---------- Function Definitions -------------------------------------------

vector<vector<bool> > readNxNMatrix (std::string adj_mat_file_name)
{
  ifstream adj_mat_file;
  adj_mat_file.open (adj_mat_file_name.c_str (), ios::in);
  if (adj_mat_file.fail ())
    {
      NS_FATAL_ERROR ("File " << adj_mat_file_name.c_str () << " not found");
    }
  vector<vector<bool> > array;
  int i = 0;
  int n_nodes = 0;

  while (!adj_mat_file.eof ())
    {
      string line;
      getline (adj_mat_file, line);
      if (line == "")
        {
          NS_LOG_WARN ("WARNING: Ignoring blank row in the array: " << i);
          break;
        }

      istringstream iss (line);
      bool element;
      vector<bool> row;
      int j = 0;

      while (iss >> element)
        {
          row.push_back (element);
          j++;
        }

      if (i == 0)
        {
          n_nodes = j;
          NS_LOG_UNCOND("n_nodes "<<n_nodes);
        }

      if (j != n_nodes )
        {
          NS_LOG_ERROR ("ERROR: Number of elements in line " << i << ": " << j << " not equal to number of elements in line 0: " << n_nodes);
          NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix "<<i<<"   "<<j);
        }
      else
        {
          array.push_back (row);
        }
      i++;
    }

  if (i != n_nodes)
    {
      NS_LOG_ERROR ("There are " << i << " rows and " << n_nodes << " columns.");
      NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
    }

  adj_mat_file.close ();
  return array;

}

vector<int> readOverlayMatrix(std::string overlay_file_name)
{
  ifstream adj_mat_file;
  adj_mat_file.open (overlay_file_name.c_str (), ios::in);
  if (adj_mat_file.fail ())
    {
      NS_FATAL_ERROR ("File " << overlay_file_name.c_str () << " not found");
    }
  vector<int> array;
  int i = 0;
  //int n_nodes = 0;

  while (!adj_mat_file.eof ())
    {
      string line;
      getline (adj_mat_file, line);
      if (line == "")
        {
          NS_LOG_WARN ("WARNING: Ignoring blank row in the array: " << i);
          break;
        }

      istringstream iss (line);
      int element;
      int j = 0;

      while (iss >> element)
        {
          array.push_back (element);
          j++;
        }

      //if (i == 0)
      //  {
      //    n_nodes = j;
      //  }
      //
      //if (j != n_nodes )
      //  {
      //    NS_LOG_ERROR ("ERROR: Number of elements in line " << i << ": " << j << " not equal to number of elements in line 0: " << n_nodes);
      //    NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
      //  }
      //else
      //  {
      //    //array.push_back (row);
      //  }
      i++;
    }

  //if (i != n_nodes)
  //  {
  //    NS_LOG_ERROR ("There are " << i << " rows and " << n_nodes << " columns.");
  //    NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
  //  }

  adj_mat_file.close ();
  return array;

}

vector<vector<double> > readOptRejectedFile (std::string opt_rejected_file_name)
{
  ifstream adj_mat_file;
  adj_mat_file.open (opt_rejected_file_name.c_str (), ios::in);
  if (adj_mat_file.fail ())
    {
      NS_FATAL_ERROR ("File " << opt_rejected_file_name.c_str () << " not found");
    }
  vector<vector<double> > array;
  int i = 0;
  int n_nodes = 0;

  while (!adj_mat_file.eof ())
    {
      string line;
      getline (adj_mat_file, line);
      if (line == "")
        {
          NS_LOG_UNCOND ("WARNING: Ignoring blank row in the array: " << i);
          break;
        }

      istringstream iss (line);
      double element;
      vector<double> row;
      int j = 0;

      while (iss >> element)
        {
          row.push_back (element);
          j++;
        }

      if (i == 0)
        {
          n_nodes = j;
        }

      if (j != n_nodes )
        {
          NS_LOG_ERROR ("ERROR: Number of elements in line " << i << ": " << j << " not equal to number of elements in line 0: " << n_nodes);
          NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
        }
      else
        {
          array.push_back (row);
        }
      i++;
    }

  if (i != n_nodes)
    {
      NS_LOG_UNCOND ("There are " << i << " rows and " << n_nodes << " columns.");
      NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
    }

  adj_mat_file.close ();
  return array;

}

vector<vector<double> > readCordinatesFile (std::string node_coordinates_file_name)
{
  ifstream node_coordinates_file;
  node_coordinates_file.open (node_coordinates_file_name.c_str (), ios::in);
  if (node_coordinates_file.fail ())
    {
      NS_FATAL_ERROR ("File " << node_coordinates_file_name.c_str () << " not found");
    }
  vector<vector<double> > coord_array;
  int m = 0;

  while (!node_coordinates_file.eof ())
    {
      string line;
      getline (node_coordinates_file, line);

      if (line == "")
        {
          NS_LOG_WARN ("WARNING: Ignoring blank row: " << m);
          break;
        }

      istringstream iss (line);
      double coordinate;
      vector<double> row;
      int n = 0;
      while (iss >> coordinate)
        {
          row.push_back (coordinate);
          n++;
        }

      if (n != 2)
        {
          NS_LOG_ERROR ("ERROR: Number of elements at line#" << m << " is "  << n << " which is not equal to 2 for node coordinates file");
          exit (1);
        }

      else
        {
          coord_array.push_back (row);
        }
      m++;
    }
  node_coordinates_file.close ();
  return coord_array;

}

vector<vector<std::string>> readIntensityFile (std::string intensity_file_name)
{
  ifstream node_intensity_file;
  node_intensity_file.open (intensity_file_name.c_str (), ios::in);
  if (node_intensity_file.fail ())
    {
      NS_FATAL_ERROR ("File " << intensity_file_name.c_str () << " not found");
    }
  vector<vector<std::string>> intensity_array;
  int i = 0;
  int n_nodes = 0;

  while (!node_intensity_file.eof ())
    {
      string line;
      getline (node_intensity_file, line);
      if (line == "")
        {
          NS_LOG_WARN ("WARNING: Ignoring blank row in the array: " << i);
          break;
        }

      istringstream iss (line);
      std::string element;
      vector<std::string> row;
      int j = 0;

      while (iss >> element)
        {
          row.push_back (element);
          j++;
        }

      if (i == 0)
        {
          n_nodes = j;
        }

      if (j != n_nodes )
        {
          NS_LOG_ERROR ("ERROR: Number of elements in line " << i << ": " << j << " not equal to number of elements in line 0: " << n_nodes);
          NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
        }
      else
        {
          intensity_array.push_back (row);
        }
      i++;
    }

  if (i != n_nodes)
    {
      NS_LOG_ERROR ("There are " << i << " rows and " << n_nodes << " columns.");
      NS_FATAL_ERROR ("ERROR: The number of rows is not equal to the number of columns! in the adjacency matrix");
    }

  node_intensity_file.close ();
  return intensity_array;

  
}

void printMatrix (const char* description, vector<vector<bool> > array)
{
  cout << "**** Start " << description << "********" << endl;
  for (size_t m = 0; m < array.size (); m++)
    {
      for (size_t n = 0; n < array[m].size (); n++)
        {
          cout << array[m][n] << ' ';
        }
      cout << endl;
    }
  cout << "**** End " << description << "********" << endl;

}

void printCoordinateArray (const char* description, vector<vector<double> > coord_array)
{
  cout << "**** Start " << description << "********" << endl;
  for (size_t m = 0; m < coord_array.size (); m++)
    {
      for (size_t n = 0; n < coord_array[m].size (); n++)
        {
          cout << coord_array[m][n] << ' ';
        }
      cout << endl;
    }
  cout << "**** End " << description << "********" << endl;

}

void ScheduleNextTrainStep(Ptr<PacketRoutingEnv> openGym)
{
  // Simulator::Schedule (Seconds(envStepTime), &ScheduleNextTrainStep, envStepTime, openGym);
  // openGym->NotifyTrainStep(openGym);
}

void ScheduleHelloMessages(Ipv4OSPFRouting* ospf){
  NS_LOG_UNCOND("Here");
  ospf->sendHelloAll();
}



// ---------- End of Function Definitions ------------------------------------
