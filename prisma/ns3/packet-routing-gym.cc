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
#include "point-to-point-net-device.h"
#include "ns3/ppp-header.h"
#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"
#include "my-tag.h"
#include "ospf-tag.h"

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
  
PacketRoutingEnv::PacketRoutingEnv (Ptr<Node> node, uint32_t numberOfNodes, uint64_t linkRateValue, bool activateSignaling, double signPacketSize, vector<int> overlayNeighbors)
{
  NS_LOG_FUNCTION (this);
  //NetDeviceContainer m_list_p2pNetDevs = list_p2pNetDevs;
  m_packetRate = (float) linkRateValue;
  m_n_nodes = numberOfNodes;
  m_node = node;
  m_overlayNeighbors = overlayNeighbors;
  m_lastEvNumPktsInQueue = 0;
  m_lastEvNode = 0;
  m_lastEvDev_idx = 1;
  m_fwdDev_idx = 1;
  is_trainStep_flag = 0;
  m_activateSignaling = activateSignaling;
  m_signPacketSize = signPacketSize;
  for(uint32_t i=0;i<m_n_nodes;i++){
    m_lsaSeen.push_back(false);
  }
  m_lsaSeen[m_node->GetId()] = true;
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
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (num_devs-2); // first dev is not p2p
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetActionSpace: " << space);
  return space;
}

Ptr<OpenGymSpace>
PacketRoutingEnv::GetObservationSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t num_devs = m_node->GetNDevices() -1;
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
  ////NS_LOG_UNCOND(m_node->GetId()<<"     "<<m_dest);
  if (is_trainStep_flag==0){
    m_isGameOver = (m_dest==m_node->GetId());
  }
  ////NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetGameOver: " << m_isGameOver);
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
  for (uint32_t i=2 ; i<num_devs; i++){
    Ptr<NetDevice> netDev = m_node->GetDevice (i);
    // uint32_t value = GetQueueLength (m_node, i);
    uint32_t value = GetQueueLengthInBytes (m_node, i);
    
    box->AddValue(value);
  }

  //NS_LOG_UNCOND ( "Node: " << m_node->GetId() << ", MyGetObservation: " << box);
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
  //   //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetReward: " << reward);
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
    if(m_signaling==0){
      myInfo += std::to_string(m_pckt->GetUid());
    }
    else{
      //NS_LOG_UNCOND(std::to_string(m_pcktIdSign));
      myInfo += std::to_string(m_pcktIdSign);
    }

    if(m_signaling==1){
      NS_LOG_UNCOND("SIGNALING");
    }
    myInfo += ", Signaling =";
    myInfo += std::to_string(m_signaling);

    myInfo += ", NodeIdSignaled =";
    myInfo += std::to_string(m_nodeIdSign);

    myInfo += ", NNIndex =";
    myInfo += std::to_string(m_NNIndex);

    myInfo += ", segIndex =";
    myInfo += std::to_string(m_segIndex);
    
    myInfo += ", nbPktsObs =";
    uint32_t num_devs = m_node->GetNDevices();
    Ptr<OpenGymBoxContainer<int32_t> > box = CreateObject<OpenGymBoxContainer<int32_t> >(m_obs_shape);
    if (is_trainStep_flag==0){
      myInfo += std::to_string(m_dest);
      myInfo += ";";
    }
    else{
      int32_t train_reward = -1;
      myInfo += std::to_string(train_reward);
      myInfo += ";";
    }
    for (uint32_t i=2 ; i<num_devs; i++){
      Ptr<NetDevice> netDev = m_node->GetDevice (i);
      uint32_t value = GetQueueLength (m_node, i);
      // uint32_t value = GetQueueLengthInBytes (m_node, i);
      
      myInfo += std::to_string(value);
      myInfo += ";";
    }
    
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
  if(m_node->GetId()==1){
    
  }
  
  NS_LOG_FUNCTION (this);
  //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyExecuteActions: " << action );

  if (is_trainStep_flag==0){
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    m_fwdDev_idx = discrete->GetValue();

    if(m_ospfSignaling){
      for(uint32_t i = 2;i<m_node->GetNDevices();i++){
        Ptr<Packet> pckt = Create<Packet> (30);
        Ipv4Header ip_head;
        UdpHeader udp_head;
        pckt->AddHeader(udp_head);
        pckt->AddHeader(ip_head);
        pckt->AddPacketTag(m_lsaTag);
        Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(m_node->GetDevice(i));
        if(dev->GetIfIndex()!=m_recvDev->GetIfIndex()) dev->Send(pckt, m_destAddr, 0x800);
      } 
    }
  
    //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyExecuteActions: " << m_fwdDev_idx << "mdevices " << m_node->GetNDevices());
    if(m_isGameOver){
      m_fwdDev_idx = 1;
    } else if (m_fwdDev_idx < m_node->GetNDevices()-2 && m_signaling==0){
      m_pckt->AddHeader(m_udpHeader);
      if(m_node->GetId()==1 && m_fwdDev_idx==4) m_fwdDev_idx = 3;
      string string_ip= "10.2.2."+std::to_string(m_overlayNeighbors[m_fwdDev_idx]+1);
      Ipv4Address ip_dest(string_ip.c_str());
      m_ipHeader.SetDestination(ip_dest);
      m_pckt->AddHeader(m_ipHeader);

      Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
      ns3::Socket::SocketErrno sockerr;
      Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
      Ptr<Ipv4Route> route = routing->RouteOutput (m_pckt, m_ipHeader, 0, sockerr);
      Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
      NS_LOG_UNCOND("SENDING FROM NODE "<<dev->GetNode()->GetId()<<"    ND "<<dev->GetIfIndex());
      dev->Send(m_pckt, m_destAddr, 0x0800);


      if(m_activateSignaling){
        //NS_LOG_UNCOND("Sending "<<m_signPacketSize);
        Ptr<Packet> pckt = Create<Packet> (m_signPacketSize);
        Ipv4Header ip_head;
        UdpHeader udp_head;
        pckt->AddHeader(udp_head);
        pckt->AddHeader(ip_head);
        MyTag tagSmallSignaling;
        tagSmallSignaling.SetSimpleValue(0x02);
        uint64_t id = m_pckt->GetUid();
        //NS_LOG_UNCOND("SIGN "<<uint32_t(id));
        tagSmallSignaling.SetIdValue(id);
        pckt->AddPacketTag(tagSmallSignaling);


        //arrived = m_recvDev->Send(pckt, m_destAddr, 0x800);
        //if (arrived == 1){
        //    //NS_LOG_UNCOND ("Packet Successfully delivered");
        //}
        //else{
        //    //NS_LOG_UNCOND ("Packet Lost");
        //}
      
      }

      //p2p_netDev = DynamicCast<PointToPointNetDevice> (m_recvDev);
      //queue = p2p_netDev->GetQueue ();
      //backlog = (int) queue->GetNPackets();
      //NS_LOG_UNCOND("SIZE BUFFER AFTER "<<backlog);
    }
    else{
      //NS_LOG_UNCOND ("Packet Rejected");
    }
    
  }
  

  return true;
}
  


void
PacketRoutingEnv::NotifyPktRcv(Ptr<PacketRoutingEnv> entity, Ptr<NetDevice> netDev, NetDeviceContainer* nd, Ptr<const Packet> packet)
{
  //NS_LOG_UNCOND(packet->ToString());
  
  // define is train step flag
  //int test;
  //std::cin>>test;
  NS_LOG_UNCOND("SimTime: "<<Simulator::Now().GetMilliSeconds());
  NS_LOG_UNCOND("Node: "<<entity->m_node->GetId()<<"    ND: "<<netDev->GetIfIndex());
  entity->is_trainStep_flag = 0;
  entity->m_signaling = 0;
  entity->m_ospfSignaling = false;


  //define headers
  PppHeader ppp_head;
  

  Ptr<Packet> p;
  p = packet->Copy();
  
  OSPFTag tagOspf;
  p->PeekPacketTag(tagOspf);

  if(tagOspf.getType()==1){
    entity->m_signaling = 1;
    NS_LOG_UNCOND("HELLO MESSAGE");
  }
  if(tagOspf.getType()==2){
    if(!entity->m_lsaSeen[tagOspf.getLSANode()]){
      entity->m_ospfSignaling = true;
      entity->m_lsaSeen[tagOspf.getLSANode()] = true;
      entity->m_lsaTag = tagOspf;
    }
    entity->m_signaling = 1;
    
    //NS_LOG_UNCOND("LSA MESSAGE");
  }

  MyTag tagCopy;
  p->PeekPacketTag(tagCopy);
  uint8_t tagValue = tagCopy.GetSimpleValue();
  entity->m_dest = tagCopy.GetFinalDestination();
  entity->m_src = entity->m_node->GetId();

  NS_LOG_UNCOND("Destination: "<< entity->m_dest);
  NS_LOG_UNCOND(p->ToString());
  if(tagValue!=0){
    NS_LOG_UNCOND(p->ToString());
  }

  if(tagCopy.GetSimpleValue()==uint8_t(0x02)){
    entity->m_signaling=1;
    entity->m_pcktIdSign = tagCopy.GetIdValue();
    NS_LOG_UNCOND("BIG SIGNALING");
  }

  if(tagCopy.GetSimpleValue()==0x01){
    entity->m_signaling=1;
    entity->m_NNIndex = tagCopy.GetNNIndex();
    entity->m_segIndex = tagCopy.GetSegIndex();
    entity->m_nodeIdSign = tagCopy.GetNodeId();
    NS_LOG_UNCOND("SMALL SIGNALING");
  }
  


  //Remove Header
  p->RemoveHeader(ppp_head);  
  p->RemoveHeader(entity->m_ipHeader);
  p->RemoveHeader(entity->m_udpHeader);
  entity->m_pckt = p->Copy();

  if(entity->m_ipHeader.GetProtocol()==0x06){
    return ;
  }


  //Get Size
  entity->m_size = p->GetSize();
  
  // Get start Time
  if(tagCopy.GetSimpleValue()==0){
    entity->m_packetStart = uint32_t(tagCopy.GetStartTime());
  }
  

  
  

  //Destination and Src
  entity->m_destAddr = Mac48Address ("ff:ff:ff:ff:ff:ff");


  
  entity->m_lengthType = ppp_head.GetProtocol();
  entity->m_packetsSent = 0;
  entity->m_recvDev = netDev;
  
  
  entity->Notify();
}

void
PacketRoutingEnv::NotifyTrainStep(Ptr<PacketRoutingEnv> entity)
{
  // define is train step flag
  entity->is_trainStep_flag = 1;

  // //NS_LOG_UNCOND("train step Node "<<entity->m_node->GetId());
  // //NS_LOG_UNCOND("Packet Size: "<<entity->m_size);
  // //NS_LOG_UNCOND("Dest: "<<entity->m_dest);
  // //NS_LOG_UNCOND("Src: "<<entity->m_src);
  // //NS_LOG_UNCOND("Packet id: "<<entity->m_pckt->GetUid());
  //NS_LOG_UNCOND("Sim time : "<<Simulator::Now().GetMilliSeconds());

  entity->Notify();
}
}// ns3 namespace