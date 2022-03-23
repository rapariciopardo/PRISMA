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
 */
#include "packet-routing-gym.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"

//#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>
#include <vector>



namespace ns3 {
uint32_t PacketRoutingEnv::m_n_nodes;


NS_LOG_COMPONENT_DEFINE ("PacketRoutingEnv");

NS_OBJECT_ENSURE_REGISTERED (PacketRoutingEnv);

PacketRoutingEnv::PacketRoutingEnv ()
{
  NS_LOG_FUNCTION (this);
  
  //Ptr<PointToPointNetDevice> m_array_p2pNetDevs[m_num_nodes][m_num_nodes];
  
  //std::vector<std::vector<int>> matrix(RR, vector<int>(CC);

  m_node = 0;
  m_lastEvNumPktsInQueue = 0;
  m_lastEvNode = 0;
  m_lastEvDev_idx = 1;
  m_fwdDev_idx = 1;
  is_trainStep_flag = 0;
    //m_rxPktNum = 0;
}
  
PacketRoutingEnv::PacketRoutingEnv (Ptr<Node> node, uint32_t numberOfNodes, uint64_t linkRateValue)
{
  NS_LOG_FUNCTION (this);
  //NetDeviceContainer m_list_p2pNetDevs = list_p2pNetDevs;
  m_packetRate = (float) linkRateValue;
  m_n_nodes = numberOfNodes;
  m_node = node;
  m_lastEvNumPktsInQueue = 0;
  m_lastEvNode = 0;
  m_lastEvDev_idx = 1;
  m_fwdDev_idx = 1;
  is_trainStep_flag = 0;
  //m_rxPktNum = 0;
}

PacketRoutingEnv::PacketRoutingEnv (Time stepTime, Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);
  //NetDeviceContainer m_list_p2pNetDevs = list_p2pNetDevs;
  
  m_node = node;
  m_lastEvNumPktsInQueue = 0;
  m_lastEvNode = 0;
  m_lastEvDev_idx = 1;
  m_fwdDev_idx = 1;
  //m_rxPktNum = 0;
  m_interval = stepTime;
  is_trainStep_flag = 0;
  //Simulator::Schedule (Seconds(0.0), &PacketRoutingEnv::ScheduleNextStateRead, this);
}

void
PacketRoutingEnv::ScheduleNextStateRead ()
{
  NS_LOG_FUNCTION (this);
  Simulator::Schedule (m_interval, &PacketRoutingEnv::ScheduleNextStateRead, this);
  //Notify();
}

PacketRoutingEnv::~PacketRoutingEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
PacketRoutingEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("PacketRoutingEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<PacketRoutingEnv> ()
  ;
  return tid;
}

void
PacketRoutingEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<OpenGymSpace>
PacketRoutingEnv::GetActionSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t num_devs = m_node->GetNDevices();
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (num_devs-1); // first dev is not p2p
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetActionSpace: " << space);
  return space;
}

Ptr<OpenGymSpace>
PacketRoutingEnv::GetObservationSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t num_devs = m_node->GetNDevices();
  uint32_t low = 0;
  uint32_t high = 100000; // max buffer size --> to change depending on actual value (access to defaul sim param)
  m_obs_shape = {num_devs,}; // Destination Node + (num_devs - 1) interfaces for other nodes
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, m_obs_shape, dtype);
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetObservationSpace: " << space);
  return space;
}

bool
PacketRoutingEnv::GetGameOver()
{
  NS_LOG_FUNCTION (this);
  m_isGameOver = false;
  //NS_LOG_UNCOND(m_node->GetId()<<"     "<<m_dest);
  if (is_trainStep_flag==0){
    m_isGameOver = (m_dest==m_node->GetId());
  }
  //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetGameOver: " << m_isGameOver);
  return m_isGameOver;
}

uint32_t
PacketRoutingEnv::GetQueueLength(Ptr<Node> node, uint32_t netDev_idx)
{
  Ptr<NetDevice> netDev = node->GetDevice (netDev_idx);
  Ptr<PointToPointNetDevice> p2p_netDev = DynamicCast<PointToPointNetDevice> (netDev);
  Ptr<Queue<Packet> > queue = p2p_netDev->GetQueue ();
  uint32_t backlog = (int) queue->GetNPackets();
  return backlog;
}
uint32_t
PacketRoutingEnv::GetQueueLengthInBytes(Ptr<Node> node, uint32_t netDev_idx)
{
  Ptr<NetDevice> netDev = node->GetDevice (netDev_idx);
  Ptr<PointToPointNetDevice> p2p_netDev = DynamicCast<PointToPointNetDevice> (netDev);
  Ptr<Queue<Packet> > queue = p2p_netDev->GetQueue ();
  uint32_t backlog = (int) queue->GetNBytes();
  return backlog;
}

Ptr<OpenGymDataContainer>
PacketRoutingEnv::GetObservation()
{

  uint32_t num_devs = m_node->GetNDevices();
  Ptr<OpenGymBoxContainer<int32_t> > box = CreateObject<OpenGymBoxContainer<int32_t> >(m_obs_shape);
  if (is_trainStep_flag==0){
    box->AddValue(m_dest);
  }
  else{
    int32_t train_reward = -1;
    box->AddValue(train_reward);
  }
  
  for (uint32_t i=1 ; i<num_devs; i++){
    Ptr<NetDevice> netDev = m_node->GetDevice (i);
    uint32_t value = GetQueueLength (m_node, i);
    // uint32_t value = GetQueueLengthInBytes (m_node, i);
    
    box->AddValue(value);
  }

  NS_LOG_UNCOND ( "Node: " << m_node->GetId() << ", MyGetObservation: " << box);
  return box;
}

float
PacketRoutingEnv::GetReward()
{
  // if (is_trainStep_flag==0){
  
  //   uint32_t value = GetQueueLengthInBytes(m_node, m_fwdDev_idx);

  //   float transmission_time = (m_size*8)/m_packetRate;
  //   float waiting_time = (float)(value*8)/m_packetRate;
  //   float reward = transmission_time + waiting_time;
  //   NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetReward: " << reward);
  //   return reward;
  // }
  // else{
  return 1;
  // }
}

std::string
PacketRoutingEnv::GetExtraInfo()
{
  //NS_LOG_FUNCTION (this);
  if (is_trainStep_flag==0){
    std::string myInfo = "End to End Delay=";
    myInfo += std::to_string(Simulator::Now().GetMilliSeconds()-m_packetStart);

    myInfo += ", Packets sent =";
    myInfo += std::to_string(m_packetsSent);
    
    myInfo += ", src Node =";
    myInfo += std::to_string(m_src);

    myInfo += ", Packet Size=";
    myInfo += std::to_string(m_size);
    
    myInfo += ", Current sim time =";
    myInfo += std::to_string(Simulator::Now().GetSeconds());
    
    myInfo += ", Pkt ID =";
    myInfo += std::to_string(m_pckt->GetUid());

    return myInfo;
  }
  return "";
  // myInfo += ", Is train step =";
  // myInfo += std::to_string(is_trainStep_flag);
   
  // myInfo += ", m_lastEvNode =";
  // myInfo += std::to_string(m_lastEvNode);
  // myInfo += ", m_fwdDev_idx =";
  // myInfo += std::to_string(m_pckt->GetUid());
  // myInfo += ", m_dest =";
  // myInfo += std::to_string(m_dest);
}

bool
PacketRoutingEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyExecuteActions: " << action );

  if (is_trainStep_flag==0){
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    m_fwdDev_idx = discrete->GetValue()+1;
    if(m_isGameOver){
      NS_LOG_UNCOND("Packet arrived to destination");
      m_fwdDev_idx = 0;
    }
  
    NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyExecuteActions: " << m_fwdDev_idx);
    Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(m_node->GetDevice(m_fwdDev_idx));
    bool arrived;
    arrived=dev->Send(m_pckt, m_destAddr, 0x0800);
    if (arrived == 1){
        NS_LOG_UNCOND ("Packet Successfully delivered");
    }
    else{
        NS_LOG_UNCOND ("Packet Lost");
    }
  }
  

  return true;
}
  


void
PacketRoutingEnv::NotifyPktRcv(Ptr<PacketRoutingEnv> entity, int* counter_packets_sent, NetDeviceContainer* nd, Ptr<const Packet> packet)
{
  // define is train step flag
  entity->is_trainStep_flag = 0;

  PppHeader ppp_head;
  Ipv4Header ip_head;
  UdpHeader udp_head;
  
  Ptr<Packet> p = packet->Copy();

  NS_LOG_UNCOND("-------------------------------------------------------------");
  NS_LOG_UNCOND("Node "<<entity->m_node->GetId());

  
  //Remove Header
  p->RemoveHeader(ppp_head);
  entity->m_pckt = p->Copy();
  p->RemoveHeader(ip_head);
  p->RemoveHeader(udp_head);

  //Get Size
  entity->m_size = p->GetSize();
  
  // Get start Time
  uint8_t *buffer = new uint8_t [p->GetSize ()];
  p->CopyData(buffer, p->GetSize ());
  char* start_time_string = reinterpret_cast<char*>(&buffer[0]);
  std::string s = std::string(start_time_string);
  uint32_t start_time_int = (uint32_t) std::atoi(start_time_string);
  entity->m_packetStart = start_time_int;

  
  //NS_LOG_UNCOND("PPP--------------------------");
  //NS_LOG_UNCOND(ppp_head);
  //NS_LOG_UNCOND("IP--------------------------");
  //NS_LOG_UNCOND(ip_head);
  //NS_LOG_UNCOND("UDP--------------------------");
  //NS_LOG_UNCOND(udp_head);

  //Destination and Src

  for(uint32_t i = 0;i<nd->GetN();i++){
    Ptr<NetDevice> dev = nd->Get(i);
    Ptr<Node> n = dev->GetNode();
    Ptr<Ipv4> ipv4 = n->GetObject<Ipv4> ();
    Ipv4InterfaceAddress ipv4_int_addr = ipv4->GetAddress (1, 0);
    Ipv4Address ip_addr = ipv4_int_addr.GetLocal ();
    entity->m_destAddr = dev->GetBroadcast();

    if(ip_addr==ip_head.GetSource()){
      entity->m_srcAddr = dev->GetAddress();
      entity->m_src = dev->GetNode()->GetId()-m_n_nodes;
    }

    if(ip_addr == ip_head.GetDestination()){
      entity->m_dest = dev->GetNode()->GetId()-m_n_nodes;
    }  
  }
  entity->m_lengthType = ppp_head.GetProtocol();
  entity->m_packetsSent = *counter_packets_sent;
  
  NS_LOG_UNCOND("Packet Size: "<<entity->m_size);
  NS_LOG_UNCOND("Dest: "<<entity->m_dest);
  NS_LOG_UNCOND("Src: "<<entity->m_src);
  NS_LOG_UNCOND("pkt start: "<<entity->m_packetStart);
  NS_LOG_UNCOND("Dest IP Addr: "<<ip_head.GetDestination());
  NS_LOG_UNCOND("Src IP Addr: "<<ip_head.GetSource());
  NS_LOG_UNCOND("Packet id: "<<entity->m_pckt->GetUid());
  NS_LOG_UNCOND("Sim time : "<<Simulator::Now().GetSeconds());
  entity->Notify();
}

void
PacketRoutingEnv::NotifyTrainStep(Ptr<PacketRoutingEnv> entity)
{
  // define is train step flag
  entity->is_trainStep_flag = 1;
  NS_LOG_UNCOND("-------------------------------------------------------------");
  // NS_LOG_UNCOND("train step Node "<<entity->m_node->GetId());
  // NS_LOG_UNCOND("Packet Size: "<<entity->m_size);
  // NS_LOG_UNCOND("Dest: "<<entity->m_dest);
  // NS_LOG_UNCOND("Src: "<<entity->m_src);
  // NS_LOG_UNCOND("Packet id: "<<entity->m_pckt->GetUid());
  NS_LOG_UNCOND("Sim time : "<<Simulator::Now().GetMilliSeconds());

  entity->Notify();
}
}// ns3 namespace