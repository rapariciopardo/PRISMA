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
#include "ospf-signaling-app-helper.h"
#include "tcp-application.h"
#include "ns3/global-route-manager.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/assert.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "packet-routing-gym.h"
#include "ipv4-ospf-routing.h"
#include "conf-loader.h"


using namespace std;
using namespace ns3;




//NS_LOG_COMPONENT_DEFINE ("OpenGym");

// ---------- Prototypes ------------------------------------------------------

vector<vector<bool> > readNxNMatrix (std::string adj_mat_file_name);
vector<vector<double> > readCordinatesFile (std::string node_coordinates_file_name);
vector<vector<std::string>> readIntensityFile(std::string intensity_file_name);
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
  
  bool eventBasedEnv = true;
  double load_factor = 0.01; // scaling applied to the traffic matrix
  std::string MaxBufferLength ("30p");
  std::string LinkRate ("500Kbps");
  std::string LinkDelay ("2ms");
  uint32_t AvgPacketSize = 512 ; //—> If you want to change the by-default 512 packet size

  std::string adj_mat_file_name ("scratch/prisma/examples/abilene/adjacency_matrix.txt");
  std::string overlay_mat_file_name ("scratch/prisma/examples/abilene/overlay_matrix.txt");
  std::string node_coordinates_file_name ("scratch/prisma/examples/abilene/node_coordinates.txt");
  std::string node_intensity_file_name("scratch/prisma/examples/abilene/node_intensity.txt");

  
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



  srand ( (unsigned)time ( NULL ) );   // generate different seed each time

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



  // Optionally display 2-dimensional adjacency matrix (Adj_Matrix) array
  // printMatrix (adj_mat_file_name.c_str (),Adj_Matrix);

  // ---------- End of Read Adjacency Matrix ---------------------------------

  // ---------- Read Node Coordinates File -----------------------------------



  vector<vector<double> > coord_array;
  coord_array = readCordinatesFile (node_coordinates_file_name);

  vector<vector<std::string>> Traff_Matrix;
  Traff_Matrix = readIntensityFile (node_intensity_file_name);

  

  int n_nodes = coord_array.size ();
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
  //Creating the IP Stack Helpers
  InternetStackHelper internet;
  internet.Install(nodes_traffic);
  internet.Install(nodes_switch);

  //Parameters of signaling
  double smallSignalingSize[n_nodes] = {0.0};
  //double bigSignalingSize = 36000;
  
  if(signalingType=="ideal"){
    NS_LOG_UNCOND("SMALL SIGNALING");
    for(int i=0;i<n_nodes;i++){
      smallSignalingSize[i] = 8 + (8 * (nodes_degree[i]+1));
    }
  } else if(signalingType=="target"){
    for(int i=0;i<n_nodes;i++){
      smallSignalingSize[i] = 16;
    }
  }  

  //Creating the links
  NS_LOG_UNCOND("Creating link between switch nodes");
  
  for(int i=0;i<n_nodes;i++)
  {
    PointToPointHelper p2p;
    DataRate data_rate(LinkRate);

    p2p.SetDeviceAttribute ("DataRate", DataRateValue (10000*data_rate.GetBitRate()*nodes_degree[i]));
    p2p.SetChannelAttribute ("Delay", StringValue (LinkDelay));
    p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("500p"));    
    NetDeviceContainer n_devs = p2p.Install (NodeContainer (nodes_traffic.Get(i), nodes_switch.Get(i)));
    dev_links.push_back(n_devs);
    traffic_nd.Add(n_devs.Get(0));
    switch_nd.Add(n_devs.Get(1));
  }

  vector<tuple<int, int>> link_devs;
  Ptr<NetDevice> nds_switch [n_nodes][n_nodes];
  for (size_t i = 0; i < Adj_Matrix.size (); i++)
      {
        for (size_t j = i; j < Adj_Matrix[i].size (); j++)
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
  for(size_t i=0;i<link_devs.size();i++){
    NS_LOG_UNCOND(get<0>(link_devs[i]) << "     " << get<1>(link_devs[i])<<"    "<<switch_nd.Get(get<0>(link_devs[i]))->GetNode()->GetId()<<"     "<<switch_nd.Get(get<1>(link_devs[i]))->GetNode()->GetId());
    NS_LOG_UNCOND(switch_nd.Get(get<0>(link_devs[i]))->GetIfIndex());
  }


  //Adding Ipv4 Address to the Net Devices
  Ipv4AddressHelper address;
  address.SetBase ("10.2.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces_traffic = address.Assign (traffic_nd);

  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces_switch = address.Assign (switch_nd);

  // Create the overlay links
  vector<int> overlayNodes;
  vector<int> overlayNeighbors[n_nodes];
  bool overlayNodesChecker[n_nodes] = {0};
  for(int i=0;i<n_nodes;i++){
    for(int j=i;j<n_nodes;j++){
      if(OverlayAdj_Matrix[i][j]==1){
        if(overlayNodesChecker[i]==0){
          overlayNodes.push_back(i);
          overlayNodesChecker[i] = 1;
        }
        if(overlayNodesChecker[j]==0){
          overlayNodes.push_back(j);
          overlayNodesChecker[j] = 1;
        }
        overlayNeighbors[i].push_back(j);
        overlayNeighbors[j].push_back(i);
      }
    }
  }

  // OpenGym Env
  NS_LOG_INFO ("Setting up OpemGym Envs for each node.");
  
  std::vector<Ptr<OpenGymInterface> > myOpenGymInterfaces;
  std::vector<Ptr<PacketRoutingEnv> > packetRoutingEnvs;
  
  uint64_t linkRateValue= DataRate(LinkRate).GetBitRate();
  
  for (size_t i = 0; i < overlayNodes.size(); i++)
  {
    NS_LOG_UNCOND("Node: "<<overlayNodes[i]);
    for(int neighbor : overlayNeighbors[overlayNodes[i]]){
      NS_LOG_UNCOND(neighbor);
    }
    
    Ptr<Node> n = nodes_switch.Get (overlayNodes[i]); // ref node
    //nodeOpenGymPort = openGymPort + i;
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort + overlayNodes[i]);
     Ptr<PacketRoutingEnv> packetRoutingEnv;
    if (eventBasedEnv){
      packetRoutingEnv = CreateObject<PacketRoutingEnv> (n, n_nodes, linkRateValue, activateSignaling, smallSignalingSize[overlayNodes[i]], overlayNeighbors[overlayNodes[i]]); // event-driven step
    } else {
      packetRoutingEnv = CreateObject<PacketRoutingEnv> (Seconds(envStepTime), n); // time-driven step
    }
    packetRoutingEnv->SetOpenGymInterface(openGymInterface);
    for(size_t j = 1;j<nodes_switch.Get(overlayNodes[i])->GetNDevices();j++){
      Ptr<NetDevice> dev_switch =DynamicCast<NetDevice> (nodes_switch.Get(overlayNodes[i])->GetDevice(j)); 
      NS_LOG_UNCOND(dev_switch->GetNode()->GetId()<<"     "<<j);
      dev_switch->TraceConnectWithoutContext("MacRx", MakeBoundCallback(&PacketRoutingEnv::NotifyPktRcv, packetRoutingEnv, dev_switch, &traffic_nd));
    }
    
    myOpenGymInterfaces.push_back (openGymInterface);
    packetRoutingEnvs.push_back (packetRoutingEnv);

    
      
  }  

  

  ///////////////////////////////////////////////////////////

      
  NS_LOG_UNCOND ("Create Links Between Nodes & Connecting OpenGym entity to event sources.");
  //NetDeviceContainer list_p2pNetDevs = NetDeviceContainer();
  
  
  uint32_t linkCount = 0;
  
  NS_LOG_INFO ("Number of links in the adjacency matrix is: " << linkCount);
  //NS_LOG_INFO ("Number of all nodes is: " << nodes_switch.GetN ());

  

 

  // ---------- Create n*(n-1) CBR Flows -------------------------------------

  
  NS_LOG_INFO ("Setup CBR Traffic Sources.");

  
  
  
  uint16_t sinkPortUDP = 11;
  
  //int interface = 0;
  for (int i = 0; i < n_nodes; i++)
    {
      //interface = 0;
      for (int j = 0; j < n_nodes; j++)
        {
          if (i != j && i==0 && j==5)
            {
  
              // We needed to generate a random number (rn) to be used to eliminate
              // the artificial congestion caused by sending the packets at the
              // same time. This rn is added to AppStartTime to have the sources
              // start at different time, however they will still send at the same rate.
              
              Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
              x->SetAttribute ("Min", DoubleValue (0));
              x->SetAttribute ("Max", DoubleValue (1));
              

              Address sinkAddress;
              //Ptr<Node> n = nodes_traffic.Get (j);
              //Ptr<Ipv4> ipv4 = n->GetObject<Ipv4> ();
              //Ipv4InterfaceAddress ipv4_int_addr = ipv4->GetAddress (1, 0);
              //Ipv4Address ip_addr = ipv4_int_addr.GetLocal ();
              Ipv4Address ip_test("10.1.1.1");
              NS_LOG_UNCOND(ip_test);
              sinkAddress = InetSocketAddress (ip_test, sinkPortUDP);
              
              double rn = x->GetValue ();
              PoissonAppHelper poisson  ("ns3::UdpSocketFactory",sinkAddress);
              poisson.SetAverageRate (DataRate(round(DataRate(Traff_Matrix[i][j]).GetBitRate()*load_factor)), AvgPacketSize);
              poisson.SetUpdatable(false, updateTrafficRateTime);
              poisson.SetDestination(uint32_t (j+1));
              ApplicationContainer apps = poisson.Install (nodes_traffic.Get (i));
              apps.Start (Seconds (AppStartTime + rn));
              apps.Stop (Seconds (AppStopTime));            
              
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
  NS_LOG_UNCOND("Sent Packets: "<< counter_send);
  for (size_t i = 0; i < myOpenGymInterfaces.size(); i++)
  {
    NS_LOG_UNCOND("flags" << i);
    packetRoutingEnvs[i]->is_trainStep_flag = 1;
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
      NS_LOG_ERROR ("There are " << i << " rows and " << n_nodes << " columns.");
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
  openGym->NotifyTrainStep(openGym);
}

void ScheduleHelloMessages(Ipv4OSPFRouting* ospf){
  NS_LOG_UNCOND("Here");
  ospf->sendHelloAll();
}



// ---------- End of Function Definitions ------------------------------------
