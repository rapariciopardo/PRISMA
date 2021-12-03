/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Egemen K. Cetinkaya, Justin P. Rohrer, and Amit Dandekar
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
 * Author: Egemen K. Cetinkaya <ekc@ittc.ku.edu>
 * Author: Justin P. Rohrer    <rohrej@ittc.ku.edu>
 * Author: Amit Dandekar       <dandekar@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center
 * and
 * Department of Electrical Engineering and Computer Science
 * The University of Kansas
 * Lawrence, KS  USA
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture) and
 * by NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI)
 *
 * This program reads an upper triangular adjacency matrix (e.g. adjacency_matrix.txt) and
 * node coordinates file (e.g. node_coordinates.txt). The program also set-ups a
 * wired network topology with P2P links according to the adjacency matrix with
 * nx(n-1) CBR traffic flows, in which n is the number of nodes in the adjacency matrix.
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
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/assert.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "mygym.h"

using namespace std;
using namespace ns3;

//static std::vector<uint32_t> rxPkts;

//static void
//CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, const Address & srcAddr)
//{
//  printf("aquiiiiiiiiiiii\n");
//  rxPkts[sinkId]++;
//}


//NS_LOG_COMPONENT_DEFINE ("OpenGym");

// ---------- Prototypes ------------------------------------------------------

vector<vector<bool> > readNxNMatrix (std::string adj_mat_file_name);
vector<vector<double> > readCordinatesFile (std::string node_coordinates_file_name);
void printCoordinateArray (const char* description, vector<vector<double> > coord_array);
void printMatrix (const char* description, vector<vector<bool> > array);
void NotifyPktRecv(int i, Ptr<const Packet> packet)
  {
    NS_LOG_UNCOND("AQUI");

  }


/*void DevicePacketsInQueueTrace (uint32_t oldValue, uint32_t newValue){
    std::cout  << "Time stamp: " << Simulator::Now () << ", Context: " << this <<   ", DevicePacketsInQueue: " << oldValue << " to " << newValue << std::endl;
}*/
/*
static void DevPacketsInQueue (std::string context, uint32_t oldValue, uint32_t newValue)
{
    //uint32_t backlog = queue->GetNPackets ();
    std::size_t pos = context.find("$");      // position of "live" in str
    std::string node_dev_pair_str = context.substr (0,pos);     // get from "live" to the end
    std::cout << "Time stamp: " << Simulator::Now () << ", Context: " << node_dev_pair_str << ", DevicePacketsInQueue: " << newValue << std::endl;
}
*/
NS_LOG_COMPONENT_DEFINE ("GenericTopologyCreation");

int main (int argc, char *argv[])
{

  // ---------- Simulation Variables ------------------------------------------

  // Change the variables and file names only in this block!
  // Parameters of the environment
  uint32_t simSeed = 1;
  uint32_t openGymPort = 5555;
  uint32_t testArg = 0;
  double simTime        = 3.00; //seconds
  double envStepTime = 0.1; //seconds, ns3gym env step time interval
  
  bool eventBasedEnv = true;
  
  //Parameters of the scenario
  double SinkStartTime  = 1.0001;
  double SinkStopTime   = 9.90001;
  double AppStartTime   = 2.0001;
  double AppStopTime    = 9.80001;
    
  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("eventBasedEnv", "Whether steps should be event or time based. Default: true", eventBasedEnv);
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simTime);
  cmd.AddValue ("stepTime", "Gym Env step time in seconds. Default: 0.1s", envStepTime);
  cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.Parse (argc, argv);
    
  NS_LOG_UNCOND("Ns3Env parameters:");
  NS_LOG_UNCOND("--simulationTime: " << simTime);
  NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND("--seed: " << simSeed);
  NS_LOG_UNCOND("--testArg: " << testArg);
    
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);
  
  LogComponentEnable ("GenericTopologyCreation", LOG_LEVEL_INFO);

  //std::string AppPacketRate ("40Kbps");
  std::string AppPacketRate ("500Kbps");
  Config::SetDefault  ("ns3::OnOffApplication::PacketSize",StringValue ("1000"));
  Config::SetDefault ("ns3::OnOffApplication::DataRate",  StringValue (AppPacketRate));
  std::string LinkRate ("10Mbps");
  std::string LinkDelay ("2ms");
  //  DropTailQueue::MaxPackets affects the # of dropped packets, default value:100
  //  Config::SetDefault ("ns3::DropTailQueue::MaxPackets", UintegerValue (1000));

  srand ( (unsigned)time ( NULL ) );   // generate different seed each time

  std::string tr_name ("n-node-ppp.tr");
  std::string pcap_name ("n-node-ppp");
  std::string flow_name ("n-node-ppp.xml");
  std::string anim_name ("n-node-ppp.anim.xml");

  std::string adj_mat_file_name ("scratch/my_network/adjacency_matrix.txt");
  std::string node_coordinates_file_name ("scratch/my_network/node_coordinates.txt");
  //std::string adj_mat_file_name ("input/adjacency_matrix.txt");
  //std::string node_coordinates_file_name ("input/node_coordinates.txt");

  //CommandLine cmd;
  //cmd.Parse (argc, argv);
  

  // ---------- End of Simulation Variables ----------------------------------

  // ---------- Read Adjacency Matrix ----------------------------------------

  vector<vector<bool> > Adj_Matrix;
  Adj_Matrix = readNxNMatrix (adj_mat_file_name);

  // Optionally display 2-dimensional adjacency matrix (Adj_Matrix) array
  // printMatrix (adj_mat_file_name.c_str (),Adj_Matrix);

  // ---------- End of Read Adjacency Matrix ---------------------------------

  // ---------- Read Node Coordinates File -----------------------------------

  vector<vector<double> > coord_array;
  coord_array = readCordinatesFile (node_coordinates_file_name);

  // Optionally display node co-ordinates file
  // printCoordinateArray (node_coordinates_file_name.c_str (),coord_array);

  int n_nodes = coord_array.size ();
  int matrixDimension = Adj_Matrix.size ();

  if (matrixDimension != n_nodes)
    {
      NS_FATAL_ERROR ("The number of lines in coordinate file is: " << n_nodes << " not equal to the number of nodes in adjacency matrix size " << matrixDimension);
    }

  // ---------- End of Read Node Coordinates File ----------------------------

  // ---------- Network Setup ------------------------------------------------

  NS_LOG_INFO ("Create Nodes.");

  NodeContainer nodes_traffic;   // Declare nodes objects
  nodes_traffic.Create (n_nodes);

  NodeContainer nodes_switch;
  nodes_switch.Create(n_nodes);

  NS_LOG_INFO ("Create P2P Link Attributes.");

  //PointToPointHelper p2p;
  CsmaHelper p2p;
  p2p.SetChannelAttribute ("DataRate", DataRateValue (LinkRate));
  p2p.SetChannelAttribute ("Delay", StringValue (LinkDelay));



  Ptr<Node> n0 = CreateObject<Node> ();
  Ptr<Node> n1 = CreateObject<Node> ();

  Ptr<Node> bridge1 = CreateObject<Node> ();


  
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));

  NetDeviceContainer topLanDevices;
  NetDeviceContainer topBridgeDevices;

  NetDeviceContainer link = csma.Install (NodeContainer (n0, bridge1));
  topLanDevices.Add (link.Get (0));
  topBridgeDevices.Add (link.Get (1));

  NetDeviceContainer link2 = csma.Install (NodeContainer (n1, bridge1));
  topLanDevices.Add (link2.Get (0));
  topBridgeDevices.Add (link2.Get (1));

  for(uint32_t i=0;i<topBridgeDevices.GetN();i++){
    Ptr<CsmaNetDevice> nd_new = DynamicCast<CsmaNetDevice>(topBridgeDevices.Get(i)); 
    nd_new->TraceConnectWithoutContext("MacRx", MakeBoundCallback(NotifyPktRecv, i));
  }

  NodeContainer routerNodes (n0, n1);
  InternetStackHelper internet_new;
  internet_new.Install (routerNodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (topLanDevices);

  uint16_t port_new = 9;   // Discard port (RFC 863)
  
  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                      Address (InetSocketAddress (Ipv4Address ("10.1.1.3"), port_new)));
  onoff.SetConstantRate (DataRate ("500kb/s"));
  
  ApplicationContainer app = onoff.Install (n0);
  // Start the application
  app.Start (Seconds (1.0));
  app.Stop (Seconds (10.0));
    
  // Create an optional packet sink to receive these packets
  PacketSinkHelper sinknew ("ns3::UdpSocketFactory",
                         Address (InetSocketAddress (Ipv4Address::GetAny (), port_new)));
  ApplicationContainer sink1 = sinknew.Install (n1);
  sink1.Start (Seconds (1.0));
  //p2p.SetDeviceAttribute ("DataRate", StringValue (LinkRate));
  //p2p.SetChannelAttribute ("Delay", StringValue (LinkDelay));

  NS_LOG_INFO ("Install Internet Stack to Nodes.");

  InternetStackHelper internet;
  internet.Install (nodes_traffic);

  NS_LOG_INFO ("Assign Addresses to Nodes.");

  Ipv4AddressHelper ipv4_n;
  ipv4_n.SetBase ("10.0.0.0", "255.255.255.252");

  uint16_t port = 9;


  // OpenGym Env
  NS_LOG_INFO ("Setting up OpemGym Envs for each node.");
  
  std::vector<Ptr<OpenGymInterface> > myOpenGymInterfaces;
  std::vector<Ptr<MyGymEnv> > myGymEnvs;



  for (int i = 0; i < n_nodes; i++)
  {
    Ptr<Node> n = nodes_switch.Get (i); // ref node
    //nodeOpenGymPort = openGymPort + i;
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort + i);
     Ptr<MyGymEnv> myGymEnv;
    if (eventBasedEnv){
      myGymEnv = CreateObject<MyGymEnv> (n); // event-driven step
    } else {
      myGymEnv = CreateObject<MyGymEnv> (Seconds(envStepTime), n); // time-driven step
    }
    myGymEnv->SetOpenGymInterface(openGymInterface);
    
    myOpenGymInterfaces.push_back (openGymInterface);
    myGymEnvs.push_back (myGymEnv);
  }

  NS_LOG_UNCOND("Creating link between switch nodes");
  NetDeviceContainer traffic_nd;
  NetDeviceContainer switch_nd;
  for(int i=0;i<n_nodes;i++)
  {
    NodeContainer n_links = NodeContainer (nodes_traffic.Get(i), nodes_switch.Get(i));
    NetDeviceContainer n_devs = p2p.Install (n_links);

    


    //NetDeviceContainer traffic_nd;
    //traffic_nd.Add(n_devs.Get(0));
    //NetDeviceContainer switch_nd;
    //switch_nd.Add(n_devs.Get(1));
    Ptr<NetDevice> dev_traffic = n_devs.Get(0);
    Ptr<CsmaNetDevice> dev_switch =DynamicCast<CsmaNetDevice> (n_devs.Get(1)); //CreateObject<CsmaNetDevice> ();
    
    ipv4_n.Assign (NetDeviceContainer(dev_traffic));
    ipv4_n.NewNetwork ();

    traffic_nd.Add(dev_traffic);
    switch_nd.Add(dev_switch);
    
    //NS_LOG_UNCOND( n_devs.GetN());
    //Ptr<NetDevice> nd = n_devs.Get(1);
    //Ptr<CsmaNetDevice> nd =DynamicCast<CsmaNetDevice> (switch_nd.Get(0)); //CreateObject<CsmaNetDevice> ();
    NS_LOG_UNCOND(dev_switch);
    //nd->SetAddress (Mac48Address::Allocate ());
    //NS_LOG_UNCOND(nd->GetAddress());
    //Ptr<Node> n = nodes_switch.Get(i); // ref node
    //n->AddDevice (nd);
    //nd->SetQueue (CreateObject<DropTailQueue<Packet> > ());
    dev_switch->TraceConnectWithoutContext("MacRx", MakeBoundCallback(&MyGymEnv::NotifyPktRcv, i));

  }
  NS_LOG_UNCOND("Number of traffic devices: "<<traffic_nd.GetN());
  //ipv4_n.Assign (traffic_nd);
  //ipv4_n.NewNetwork ();

  
      
  NS_LOG_INFO ("Create Links Between Nodes & Connecting OpenGym entity to event sources.");
  //NetDeviceContainer list_p2pNetDevs = NetDeviceContainer();
  
  
  uint32_t linkCount = 0;
  for (size_t i = 0; i < Adj_Matrix.size (); i++)
    {
      for (size_t j = 0; j < Adj_Matrix[i].size (); j++)
        {

          if (Adj_Matrix[i][j] == 1)
            {
              //NodeContainer n_links = NodeContainer (nodes.Get (i), nodes.Get (j));
              //NetDeviceContainer n_devs = p2p.Install (n_links);
              //ipv4_n.Assign (n_devs);
              //ipv4_n.NewNetwork ();
              //linkCount++;
              
              //list_p2pNetDevs.Add(n_devs);
              //uint32_t nDevices = n_devs.GetN ();

              //Ptr<CsmaNetDevice> dev1 = CreateObject<CsmaNetDevice> ();
              //dev1->SetAddress (Mac48Address::Allocate ());
              //Ptr<Node> n1 = nodes_switch.Get(i); // ref node
              //n1->AddDevice (dev1);
              //dev1->SetQueue (CreateObject<DropTailQueue<Packet> > ());
              //dev1->TraceConnectWithoutContext("MacPromiscRx", MakeBoundCallback(&MyGymEnv::NotifyPktRcv, i));
              //
              //
              //Ptr<CsmaNetDevice> dev2 = CreateObject<CsmaNetDevice> ();
              //dev2->SetAddress (Mac48Address::Allocate ());
              //Ptr<Node> n2 = nodes_switch.Get (j); // ref node
              //n2->AddDevice (dev2);
              //dev2->SetQueue (CreateObject<DropTailQueue<Packet> > ());
              //dev2->TraceConnectWithoutContext("MacPromiscRx", MakeBoundCallback(&MyGymEnv::NotifyPktRcv, j));


              NS_LOG_INFO("nDevivices: "<<2);
              for (uint32_t k = 0; k < 2; k++)
              {

                  //Ptr<NetDevice> nd = n_devs.Get (k);
                  //Ptr<PointToPointNetDevice> ptpnd = DynamicCast<PointToPointNetDevice> (nd);
                  //Ptr<Queue<Packet> > queue = ptpnd->GetQueue ();
                  //NS_LOG_INFO ("k: " << k << " ");
                  //NS_LOG_INFO ("i: " << i << " ");
                  //NS_LOG_INFO ("j: " << j << " ");
                  //size_t nodeDev_idx = (size_t)((ptpnd->GetNode())->GetId());
                  //NS_LOG_INFO ("starting node: " << nodeDev_idx << " ");
                  //if (eventBasedEnv){
                  //  //pktSink->TraceConnectWithoutContext ("Rx", MakeBoundCallback (&MyGymEnv::NotifyPktInQueueEvent, myGymEnvs[i], ptpnd));
                  //  //queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::NotifyPktInQueueEvent, myGymEnvs[nodeDev_idx], ptpnd)); // event-driven step
                  //}
                  //else{
                  //  queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::CountPktInQueueEvent, myGymEnvs[nodeDev_idx], ptpnd)); // time-driven step
                  //}
/*
                if (k) { // j-->i
                  if (eventBasedEnv){
                    queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::NotifyPktInQueueEvent, myGymEnvs[j], ptpnd)); // event-driven step
                  }
                  else{
                    queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::CountPktInQueueEvent, myGymEnvs[j], ptpnd)); // time-driven step
                  }
                }
                else{ // i-->j
                  if (eventBasedEnv){
                    queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::NotifyPktInQueueEvent, myGymEnvs[i], ptpnd)); // event-driven step
                  }
                  else{
                    queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::CountPktInQueueEvent, myGymEnvs[i], ptpnd)); // time-driven step
                  }
                }*/
                  //queue->TraceConnectWithoutContext ("PacketsInQueue", MakeCallback (&DevicePacketsInQueueTrace));
              }
              NS_LOG_INFO ("matrix element [" << i << "][" << j << "] is 1");
            }
          else
            {
              NS_LOG_INFO ("matrix element [" << i << "][" << j << "] is 0");
            }
        }
    }
  for (int i = 0; i < n_nodes; ++i)
    {
      //MyGymEnv::m_rxPkts.push_back(0);
      //Ptr<PacketSink> pktSink = DynamicCast<PacketSink>(apps_sink.Get((uint32_t)i));
      //Config::Connect("/NodeList/$ns3::Ipv4L3Protocol/Rx", MakeBoundCallback (&MyGymEnv::CountRxPkts, i));
      //Ptr<CsmaNetDevice> deviceA = CreateObject<CsmaNetDevice> ();
      //deviceA->SetAddress (Mac48Address::Allocate ());
      //nA->AddDevice (deviceA);
      //deviceA->SetQueue (CreateObject<DropTailQueue<Packet> > ());
    }
  NS_LOG_INFO ("Number of links in the adjacency matrix is: " << linkCount);
  NS_LOG_INFO ("Number of all nodes is: " << nodes_switch.GetN ());

  NS_LOG_INFO ("Initialize Global Routing.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // ---------- End of Network Set-up ----------------------------------------

  // ---------- Allocate Node Positions --------------------------------------

  /*NS_LOG_INFO ("Allocate Positions to Nodes.");

  MobilityHelper mobility_n;
  Ptr<ListPositionAllocator> positionAlloc_n = CreateObject<ListPositionAllocator> ();

  for (size_t m = 0; m < coord_array.size (); m++)
    {
      positionAlloc_n->Add (Vector (coord_array[m][0], coord_array[m][1], 0));
      Ptr<Node> n0 = nodes_switch.Get (m);
      Ptr<ConstantPositionMobilityModel> nLoc =  n0->GetObject<ConstantPositionMobilityModel> ();
      if (nLoc == 0)
        {
          nLoc = CreateObject<ConstantPositionMobilityModel> ();
          n0->AggregateObject (nLoc);
        }
      // y-coordinates are negated for correct display in NetAnim
      // NetAnim's (0,0) reference coordinates are located on upper left corner
      // by negating the y coordinates, we declare the reference (0,0) coordinate
      // to the bottom left corner
      Vector nVec (coord_array[m][0], -coord_array[m][1], 0);
      nLoc->SetPosition (nVec);

    }
  mobility_n.SetPositionAllocator (positionAlloc_n);
  mobility_n.Install (nodes_switch);*/

  // ---------- End of Allocate Node Positions -------------------------------

  // ---------- Create n*(n-1) CBR Flows -------------------------------------

  

  NS_LOG_INFO ("Setup CBR Traffic Sources.");
  
  for (int i = 0; i < n_nodes; i++)
    {
      for (int j = 0; j < n_nodes; j++)
        {
          if (i != j)
            {
  
              // We needed to generate a random number (rn) to be used to eliminate
              // the artificial congestion caused by sending the packets at the
              // same time. This rn is added to AppStartTime to have the sources
              // start at different time, however they will still send at the same rate.
              
              Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
              x->SetAttribute ("Min", DoubleValue (0));
              x->SetAttribute ("Max", DoubleValue (1));
              double rn = x->GetValue ();
              Ptr<Node> n = nodes_traffic.Get (j);
              Ptr<Ipv4> ipv4 = n->GetObject<Ipv4> ();
              Ipv4InterfaceAddress ipv4_int_addr = ipv4->GetAddress (1, 0);
              Ipv4Address ip_addr = ipv4_int_addr.GetLocal ();
              NS_LOG_UNCOND(ipv4_int_addr);
              OnOffHelper onoff ("ns3::UdpSocketFactory", InetSocketAddress (ip_addr, port)); // traffic flows from node[i] to node[j]
              //PacketSinkHelper onoff ("ns3::UdpSocketFactory", InetSocketAddress (ip_addr, port)); // traffic flows from node[i] to node[j]              
              onoff.SetConstantRate (DataRate (AppPacketRate));
              ApplicationContainer apps = onoff.Install (nodes_traffic.Get (i));  // traffic sources are installed on all nodes
              apps.Start (Seconds (AppStartTime + rn));
              apps.Stop (Seconds (AppStopTime));
            }
        }
    }
  NS_LOG_INFO ("Setup Packet Sinks.");

  Ipv4Address sinkAddr = Ipv4Address::GetAny();
  NS_LOG_UNCOND(sinkAddr);
  PacketSinkHelper sink ("ns3::UdpSocketFactory", InetSocketAddress (sinkAddr, port));   
  ApplicationContainer apps_sink;  
  for (int i = 0; i < n_nodes; i++)
    {
      //apps_sink = sink.Install (nodes.Get (i));   // sink is installed on all nodes
      sink.SetAttribute ("Protocol", TypeIdValue (UdpSocketFactory::GetTypeId ()));
      apps_sink.Add (sink.Install (nodes_traffic.Get(i)));
    }
  apps_sink.Start (Seconds (SinkStartTime));
  apps_sink.Stop (Seconds (SinkStopTime));
  
  
  // ---------- End of Create n*(n-1) CBR Flows ------------------------------


  // ---------- Simulation Monitoring ----------------------------------------

  NS_LOG_INFO ("Configure Tracing.");
//  string probeType = "ns3::Uinteger32Probe";
  //string tracePath = "/NodeList/*/DeviceList/*/$ns3::PointToPointNetDevice/TxQueue/PacketsInQueue";
  //Config::Connect (tracePath, MakeCallback (&DevPacketsInQueue));
  //Config::Connect(tracePath, MakeBoundCallback (&MyGymEnv::CountPktInQueueEvent, myGymEnv) );
  
//
//  // Use FileHelper to write out the PacketsInQueue count over time
//  FileHelper fileHelper;
//
//  // Configure the file to be written, and the formatting of output data.
//  fileHelper.ConfigureFile ("my_mat-topology-PacketsInQueue", FileAggregator::FORMATTED);
//
//  // Set the labels for this formatted output file.
//  fileHelper.Set2dFormat ("Time (Seconds) = %.3e\t Packets In Queue = %.0f");
//
//  // Specify the probe type, probe path (in configuration namespace), and
//  // probe output trace source ("PacketsInQueue") to write.
//  fileHelper.WriteProbe (probeType, tracePath, "Output");


  AsciiTraceHelper ascii;
  p2p.EnableAsciiAll (ascii.CreateFileStream (tr_name.c_str ()));
  // p2p.EnablePcapAll (pcap_name.c_str());

  //Ptr<FlowMonitor> flowmon;
  //FlowMonitorHelper flowmonHelper;
  //flowmon = flowmonHelper.InstallAll ();

  // Configure animator with default settings
  //AnimationInterface anim (anim_name.c_str ());

/*
  
  NS_LOG_INFO ("n: " << n << "");

  NS_LOG_INFO ("Connecting OpenGym entity to event source");
  // connect OpenGym entity to event source (node output interfaces)
  for (uint32_t j=0 ; j<n->GetNDevices(); j++)
  {
    NS_LOG_INFO ("dev_idx: " << j << "");
    Ptr<NetDevice> nd = n->GetDevice (j);
    NS_LOG_INFO ("nd: " << nd << "");
    NS_LOG_INFO ("IsPointToPoint? : " << nd->IsPointToPoint () << "");

    Ptr<PointToPointNetDevice> ptpnd = DynamicCast<PointToPointNetDevice> (nd);
    NS_LOG_INFO ("ptpnd: " << ptpnd << "");
    
    Ptr<Queue<Packet> > queue = ptpnd->GetQueue ();
    queue->TraceConnectWithoutContext ("PacketsInQueue", MakeBoundCallback (&MyGymEnv::NotifyPktInQueueEvent, myGymEnv, j));
    //queue->TraceConnectWithoutContext ("PacketsInQueue", MakeCallback (&DevicePacketsInQueueTrace));
  }*/

  
  NS_LOG_INFO ("Run Simulation.");
  NS_LOG_UNCOND ("Simulation start");
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  // flowmon->SerializeToXmlFile (flow_name.c_str(), true, true);
  NS_LOG_UNCOND ("Simulation stop");
  for (int i = 0; i < n_nodes; i++)
  {
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


// ---------- End of Function Definitions ------------------------------------
