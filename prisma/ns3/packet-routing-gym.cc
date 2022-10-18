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
#include <fstream>



namespace ns3 {
uint32_t PacketRoutingEnv::m_n_nodes;
int PacketRoutingEnv::m_packetsDeliveredGlobal;
int PacketRoutingEnv::m_packetsInjectedGlobal;
int PacketRoutingEnv::m_packetsDroppedGlobal;
int PacketRoutingEnv::m_packetsDroppedTotalGlobal;
int PacketRoutingEnv::m_packetsInjectedTotalGlobal;
int PacketRoutingEnv::m_testPacketsDroppedGlobal;
int PacketRoutingEnv::m_bytesData;
int PacketRoutingEnv::m_bytesSmallSignalling;
int PacketRoutingEnv::m_bytesBigSignalling;
int PacketRoutingEnv::m_bytesOverlaySignalingForward;
int PacketRoutingEnv::m_bytesOverlaySignalingBack;
std::vector<int> PacketRoutingEnv::m_end2endDelay;
std::vector<float> PacketRoutingEnv::m_cost;

template<typename T>
double getAverage(std::vector<T> const& v) {
  if (v.empty()) {
    return 0;
  }

  double sum = 0.0;
  for (const T &i: v) {
    sum += (double)i;
  }
  return sum / v.size();
}

template<typename T>
double getSum(std::vector<T> const& v) {
  if (v.empty()) {
    return 0;
  }

  double sum = 0.0;
  for (const T &i: v) {
    sum += (double)i;
  }
  return sum;
}




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
  //for(size_t i=2;i<m_node->GetNDevices();i++){
  //  m_node->GetDevice(i)->TraceConnectWithoutContext("MacTxDrop", MakeBoundCallback(&PacketRoutingEnv::dropPacket, this));
  //}
  m_packetsDropped = 0;
  m_packetsDelivered = 0;
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
PacketRoutingEnv::setOverlayConfig(vector<int> overlayNeighbors, bool activateOverlaySignaling, uint32_t nPacketsOverlaySignaling,  uint32_t movingAverageObsSize){
  m_overlayNeighbors = overlayNeighbors;
  m_activateOverlaySignaling = activateOverlaySignaling;
  m_nPacketsOverlaySignaling = nPacketsOverlaySignaling;
  m_movingAverageObsSize = movingAverageObsSize;
  m_countSendPackets = 0;
  for(size_t i=0;i<m_overlayNeighbors.size();i++){
    m_overlayIndex[i] = 0;
    //m_tunnelsDelay.push_back(0);
    StartingOverlayPacket start;
    start.index = 0;
    start.start_time = 0;
    m_starting_overlay_packets[i].push_back(start);
  }
}

void
PacketRoutingEnv::setNetDevicesContainer(NetDeviceContainer* nd){
  for(size_t i =0; i < nd->GetN();i++){
    if(m_node->GetId()==0) nd->Get(i)->TraceConnectWithoutContext("MacTxDrop", MakeBoundCallback(&dropPacket, this));
    m_all_nds.Add(nd->Get(i));
  }
}

void
PacketRoutingEnv::setTrainConfig(bool train){
  m_train = train;
}

void
PacketRoutingEnv::setLossPenalty(double lossPenalty){
  m_loss_penalty = lossPenalty;
}

void
PacketRoutingEnv::setPingTimeout(uint32_t maxBufferSize, uint32_t linkCapacity, uint32_t propagationDelay){
  uint32_t nextHopSize[2];
  if(m_node->GetId()==0){
    nextHopSize[0] = 2;
    nextHopSize[1] = 2;
  } else if(m_node->GetId()==7){
    nextHopSize[0] = 2;
    nextHopSize[1] = 5;
  } else if(m_node->GetId()==5){
    nextHopSize[0] = 2;
    nextHopSize[1] = 2;
  } else{
    nextHopSize[0] = 2;
    nextHopSize[1] = 5;
  }
  
  
  for(size_t i=0;i<2;i++){
    m_pingTimeout[i] = ((((float(maxBufferSize)+8.0)*8.0)/float(linkCapacity)) + 0.001*float(propagationDelay))*2*nextHopSize[i];
    NS_LOG_UNCOND("Node: "<<m_node->GetId()<<"   IF: "<<i<<"   PingTimeout: "<<m_pingTimeout[i]);
  }
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
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (m_overlayNeighbors.size()); // first dev is not p2p
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetActionSpace: " << space);
  return space;
}

Ptr<OpenGymSpace>
PacketRoutingEnv::GetObservationSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t low = 0;
  uint32_t high = 300; // max buffer size --> to change depending on actual value (access to defaul sim param)
  m_obs_shape = {uint32_t(1)+uint32_t(m_overlayNeighbors.size()),}; // Destination Node + (num_devs - 1) interfaces for other nodes
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, m_obs_shape, dtype);
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetObservationSpace: " << space);
  for(size_t i = 0;i<m_overlayNeighbors.size();i++){
    m_fwdDev_idx_overlay = i;
    m_src = m_node->GetId();
    sendOverlaySignalingUpdate(uint8_t(3));
    m_lastPingOut = Simulator::Now().GetSeconds();
  }
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
PacketRoutingEnv::getNbPacketsBuffered(){
  uint32_t sum = 0;
  for(size_t i=0;i<m_all_nds.GetN();i++){
    Ptr<NetDevice> netDev = m_all_nds.Get(i);
    Ptr<PointToPointNetDevice> p2p_netDev = DynamicCast<PointToPointNetDevice> (netDev);
    Ptr<Queue<Packet> > queue = p2p_netDev->GetQueue ();
    sum += queue->GetNPackets();
  }
  return sum;
}

void
PacketRoutingEnv::dropPacket(Ptr<PacketRoutingEnv> entity, Ptr<const Packet> packet){
  MyTag tagCopy;
  packet->PeekPacketTag(tagCopy);
  if(tagCopy.GetSimpleValue()==0){
    m_packetsDroppedTotalGlobal += 1;
    if(tagCopy.GetTrafficValable()==1){
      m_cost.push_back(entity->m_loss_penalty);
      m_packetsDroppedGlobal += 1;
      entity->m_lost_packets +=std::to_string(packet->GetUid());
      entity->m_lost_packets += ";";
    }
  }  
}

uint32_t
PacketRoutingEnv::mapOverlayNode(uint32_t underlayNode){
  uint32_t res;
  if(underlayNode==0){
    res= 0;
  }
  if (underlayNode==5){
    res= 1;
  }
  if (underlayNode==7){
    res= 2;
  }
  if (underlayNode==10){
    res= 3;
  }
  return res;
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
  Ptr<OpenGymBoxContainer<int32_t> > box = CreateObject<OpenGymBoxContainer<int32_t> >(m_obs_shape);
  
  //Adding destination to obs
  if (is_trainStep_flag==0){
    box->AddValue(mapOverlayNode(m_dest));
  }
  else{
    int32_t train_reward = -1;
    box->AddValue(train_reward);
  }

  //Preparing the config
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
  ns3::Socket::SocketErrno sockerr;
  Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
  Ptr<Packet> packet_test = Create<Packet>(20);
  
  for (size_t i=0 ; i<m_overlayNeighbors.size(); i++){
    //Getting the queue sizes. (OLD VERSION)
    string string_ip= "10.2.2."+std::to_string(m_overlayNeighbors[i]+1);
    Ipv4Address ip_test(string_ip.c_str());
    m_ipHeader.SetDestination(ip_test);
    packet_test->AddHeader(m_ipHeader);
    Ptr<Ipv4Route> route = routing->RouteOutput (packet_test, m_ipHeader, 0, sockerr);
    Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
    uint value;
    if(m_activateOverlaySignaling==0){
      //uint32_t value = GetQueueLength (m_node, i);
      value = GetQueueLengthInBytes (m_node, dev->GetIfIndex());
      //NS_LOG_UNCOND("Node: "<<m_node->GetId()<<"   Value: "<<value);
    }
    else{
      if(m_starting_overlay_packets[i].size()>=1 && m_tunnelsDelay[i].size()>=1){
        if(double(Simulator::Now().GetMilliSeconds() - m_starting_overlay_packets[i][0].start_time)/2.0 >= m_pingTimeout[i]){
          value=uint(m_pingTimeout[i]*1000.0);
          //m_tunnelsDelay[i].push_back(m_pingTimeout[i]);
          //value = getAverage(m_tunnelsDelay[i]);
          //auto it = m_starting_overlay_packets[i].begin();
          //it = m_starting_overlay_packets[i].erase(it);
        } else{
          value = getAverage(m_tunnelsDelay[i]);
        }
      } else value = getAverage(m_tunnelsDelay[i]);
    }
    //NS_LOG_UNCOND("Node: "<<m_node->GetId()<<"   i: "<<i<<"    value: "<<value);
    
    box->AddValue(value);
  }
  //if(m_signaling==0) NS_LOG_UNCOND("here " <<box);

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
PacketRoutingEnv::GetLostPackets(){
  std::string lost_packets;
  if(m_train){
    for(size_t j=0;j<m_overlayNeighbors.size();j++){
      auto it = m_packetsSent[j].begin();
      while(it != m_packetsSent[j].end()){
        //NS_LOG_UNCOND(m_tunnelsDelay[j]);
        if(Simulator::Now().GetMilliSeconds() - it->start_time>= 3000){ //100*m_tunnelsDelay[j]){
          //NS_LOG_UNCOND("----------------------------------");
          //NS_LOG_UNCOND("Start: "<<it->start_time<<"  Now: "<<Simulator::Now().GetMilliSeconds());
          //NS_LOG_UNCOND("UID: "<<it->uid);
          //NS_LOG_UNCOND("----------------------------------");
          lost_packets +=std::to_string(it->uid);
          lost_packets += ";";
          it = m_packetsSent[j].erase(it);
          m_testPacketsDroppedGlobal ++;
        }
        else{
          it++;
        }
      }
    }
    //NS_LOG_UNCOND("LOST "<<loss);
  }
  
  return lost_packets;
}

std::string
PacketRoutingEnv::GetExtraInfo()
{
  //NS_LOG_FUNCTION (this);
  if (is_trainStep_flag==0){
    std::string myInfo = "End to End Delay="; //0
    myInfo += std::to_string(Simulator::Now().GetMilliSeconds()-m_packetStart);
    myInfo += ", Packets lost ="; //1
    myInfo += GetLostPackets();
    m_lost_packets = "";
    
    myInfo += ", src Node ="; //2
    myInfo += std::to_string(mapOverlayNode(m_src));

    myInfo += ", Packet Size="; //3
    myInfo += std::to_string(m_size);
    
    myInfo += ", Current sim time ="; //4
    myInfo += std::to_string(Simulator::Now().GetSeconds());
    
    myInfo += ", Pkt ID ="; //5
    if(m_signaling==0){
      myInfo += std::to_string(m_pckt->GetUid());
    }
    else{
      //NS_LOG_UNCOND(std::to_string(m_pcktIdSign));
      myInfo += std::to_string(m_pcktIdSign);
    }

    
    myInfo += ", Signaling ="; //6
    myInfo += std::to_string(m_signaling);

    myInfo += ", NodeIdSignaled ="; //7
    myInfo += std::to_string(m_nodeIdSign);

    myInfo += ", NNIndex ="; //8
    myInfo += std::to_string(m_NNIndex);

    myInfo += ", segIndex ="; //9
    myInfo += std::to_string(m_segIndex);
    
    myInfo += ", nbPktsObs ="; //10
    Ptr<OpenGymBoxContainer<int32_t> > box = CreateObject<OpenGymBoxContainer<int32_t> >(m_obs_shape);
    if (is_trainStep_flag==0){
      myInfo += std::to_string(mapOverlayNode(m_dest));
      myInfo += ";";
    }
    else{
      int32_t train_reward = -1;
      myInfo += std::to_string(train_reward);
      myInfo += ";";
    }

    Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
    ns3::Socket::SocketErrno sockerr;
    Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
    Ptr<Packet> packet_test = Create<Packet>(20);

    for (size_t i=0 ; i<m_overlayNeighbors.size(); i++){
      string string_ip= "10.2.2."+std::to_string(m_overlayNeighbors[i]+1);
      Ipv4Address ip_test(string_ip.c_str());
      m_ipHeader.SetDestination(ip_test);
      packet_test->AddHeader(m_ipHeader);
      Ptr<Ipv4Route> route = routing->RouteOutput (packet_test, m_ipHeader, 0, sockerr);
      Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
      //uint32_t value = GetQueueLength (m_node, dev->GetIfIndex());
      uint32_t value = GetQueueLengthInBytes (m_node, dev->GetIfIndex());
      
      myInfo += std::to_string(value);
      myInfo += ";";
    }

    myInfo += ", Packets dropped ="; //11
    myInfo += std::to_string(m_packetsDroppedGlobal);

    myInfo += ", Packets delivered ="; //12
    myInfo += std::to_string(m_packetsDeliveredGlobal);

    myInfo += ", Packets injected ="; //13
    myInfo += std::to_string(m_packetsInjectedGlobal);

    myInfo += ",Packets Buffered ="; //14
    myInfo += std::to_string(getNbPacketsBuffered());

    myInfo += ", Avg End to End Delay ="; //15
    myInfo += std::to_string(getAverage(m_end2endDelay)); 

    myInfo += ", Sum End to End Delay ="; //16
    myInfo += std::to_string(getSum(m_end2endDelay)); 

    myInfo += ",Cost ="; //17
    myInfo += std::to_string(getAverage(m_cost)); 

    myInfo += ",Bytes Data ="; //18
    myInfo += std::to_string(m_bytesData);

    myInfo += ",Bytes  Bigsignaling ="; //19
    myInfo += std::to_string(m_bytesBigSignalling);

    myInfo += ",Bytes  Small signaling ="; //20
    myInfo += std::to_string(m_bytesSmallSignalling);

    myInfo += ",Bytes  Overlay signaling Forward ="; //21
    myInfo += std::to_string(m_bytesOverlaySignalingForward);

    myInfo += ",Bytes  Overlay Signaling Back ="; //22
    myInfo += std::to_string(m_bytesOverlaySignalingBack);

    myInfo += ",Total Injected Packets ="; //23
    myInfo += std::to_string(m_packetsInjectedTotalGlobal);

    myInfo += ",Total Dropped Packets ="; //24
    myInfo += std::to_string(m_packetsDroppedTotalGlobal);


   

    //NS_LOG_UNCOND(myInfo);
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

void
PacketRoutingEnv::sendOverlaySignalingUpdate(uint8_t type){
  //Define Tag
  MyTag tagSmallSignaling;

  //Define packet size
  double packetSize;
  if(type==2) packetSize=m_signPacketSize;
  if(type==3) packetSize = 8;
  if(type==4) packetSize = 8;

  //Define Packet
  Ptr<Packet> smallSignalingPckt = Create<Packet> (packetSize);
  tagSmallSignaling.SetSimpleValue(type);
  if(type==2) tagSmallSignaling.SetFinalDestination(m_lastHop);
  if(type==3) tagSmallSignaling.SetFinalDestination(m_overlayNeighbors[m_fwdDev_idx_overlay]);
  if(type==4) tagSmallSignaling.SetFinalDestination(m_lastHop);
  tagSmallSignaling.SetLastHop(m_src);

  if(type==3) tagSmallSignaling.SetOverlayIndex(m_overlayIndex[m_fwdDev_idx_overlay]);
  if(type==4) tagSmallSignaling.SetOverlayIndex(m_recvOverlayIndex);

  //Depending of the type, add info the tag
  if(type==2){
    uint64_t id = m_pckt->GetUid();
    tagSmallSignaling.SetIdValue(id);
  }
  else if(type==3){
    tagSmallSignaling.SetStartTime(uint64_t(Simulator::Now().GetMilliSeconds()));
  }
  if(type==4){
    tagSmallSignaling.SetStartTime(uint64_t(Simulator::Now().GetMilliSeconds()) - m_timeStartOverlay);
  }
  smallSignalingPckt->AddPacketTag(tagSmallSignaling);

  //Adding headers
  UdpHeader udp_head;
  smallSignalingPckt->AddHeader(udp_head);
  Ipv4Header ip_head;
  string string_ip_src= "10.2.2."+std::to_string(m_node->GetId()+1);
  Ipv4Address ip_src(string_ip_src.c_str());
  ip_head.SetSource(ip_src);
  string string_ip_dest;
  if(type==2) string_ip_dest= "10.2.2."+std::to_string(m_lastHop+1);
  if(type==3) string_ip_dest= "10.2.2."+std::to_string(m_overlayNeighbors[m_fwdDev_idx_overlay]+1);
  if(type==4) string_ip_dest= "10.2.2."+std::to_string(m_lastHop+1);
  Ipv4Address ip_dest(string_ip_dest.c_str());
  ip_head.SetDestination(ip_dest);
  if(type==2) ip_head.SetPayloadSize(m_signPacketSize+udp_head.GetSerializedSize());
  if(type==3) ip_head.SetPayloadSize(8+udp_head.GetSerializedSize());
  if(type==4) ip_head.SetPayloadSize(8+udp_head.GetSerializedSize());
  ip_head.SetProtocol(17);
  smallSignalingPckt->AddHeader(ip_head);

  //Send the sign packet
  if(type==2 || type==4) m_recvDev->Send(smallSignalingPckt, m_destAddr, 0x800);
  if(type==3){
    Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
    ns3::Socket::SocketErrno sockerr;
    Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
    Ptr<Ipv4Route> route = routing->RouteOutput (smallSignalingPckt, ip_head, 0, sockerr);
    Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
    dev->Send(smallSignalingPckt, m_destAddr, 0x800);
  }


}

bool
PacketRoutingEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  if (is_trainStep_flag==0){
    
    //Get discrete action
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    m_fwdDev_idx = discrete->GetValue();
    
    //Checking for OSPF (NOT USED)
    
    //if(m_ospfSignaling){
    //  for(uint32_t i = 2;i<m_node->GetNDevices();i++){
    //    Ptr<Packet> pckt = Create<Packet> (30);
    //    Ipv4Header ip_head;
    //    UdpHeader udp_head;
    //    pckt->AddHeader(udp_head);
    //    pckt->AddHeader(ip_head);
    //    pckt->AddPacketTag(m_lsaTag);
    //    Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(m_node->GetDevice(i));
    //    if(dev->GetIfIndex()!=m_recvDev->GetIfIndex()) dev->Send(pckt, m_destAddr, 0x800);
    //  } 
    //}

    //For Data Packets which are not in source
    if(m_signaling==0 && m_activateSignaling){
      //if the limit is reached, send the overlay signaling
      if(m_countSendPackets >=m_nPacketsOverlaySignaling && m_activateOverlaySignaling){
        for(size_t i=0;i<m_overlayNeighbors.size();i++){
          m_overlayIndex[i] += 1;
          StartingOverlayPacket start;
          start.index = m_overlayIndex[i];
          start.start_time=Simulator::Now().GetMilliSeconds();
          m_starting_overlay_packets[i].push_back(start);
          m_fwdDev_idx_overlay = i;
          sendOverlaySignalingUpdate(uint8_t(3));
        }
        m_countSendPackets = 0;
        m_pingDiffs.push_back(Simulator::Now().GetSeconds()-m_lastPingOut);
        m_lastPingOut = Simulator::Now().GetSeconds();
      }
      
      if(m_lastHop!=1000 && m_train){
        //send small signaling
        //NS_LOG_UNCOND("SEND");
        sendOverlaySignalingUpdate(uint8_t(2));
      }
    }

    if(m_isGameOver){
      //NS_LOG_UNCOND("FINAL DESTINATION!");
    } else if (m_fwdDev_idx < m_overlayNeighbors.size() && m_signaling==0){
      //if(m_node->GetId()==0 ||m_node->GetId()==7) m_fwdDev_idx = 0;
      //else m_fwdDev_idx = 1;
      //NS_LOG_UNCOND(m_fwdDev_idx);
      //Replace the updated tag
      MyTag sendingTag;
      m_pckt->PeekPacketTag(sendingTag);
      sendingTag.SetLastHop(m_node->GetId());
      m_pckt->ReplacePacketTag(sendingTag);

      //Adding Headers
      m_pckt->AddHeader(m_udpHeader);
      string string_ip= "10.2.2."+std::to_string(m_overlayNeighbors[m_fwdDev_idx]+1);
      Ipv4Address ip_dest(string_ip.c_str());
      m_ipHeader.SetDestination(ip_dest);
      m_pckt->AddHeader(m_ipHeader);

      //Discovering the output buffer based on the routing table
      //(map between overlay tunnel and netDevice)
      Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
      ns3::Socket::SocketErrno sockerr;
      Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
      Ptr<Ipv4Route> route = routing->RouteOutput (m_pckt, m_ipHeader, 0, sockerr);
      Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
      
      //Send and verify if the Packet was dropped
      StartingDataPacket start;
      start.uid = m_pckt->GetUid();
      start.start_time = Simulator::Now().GetMilliSeconds();
      m_packetsSent[m_fwdDev_idx].push_back(start);
      dev->Send(m_pckt, m_destAddr, 0x0800);
      
      m_countSendPackets += 1;
    }
    else{
      //NS_LOG_UNCOND ("Not valid");
    }
    
  }
  

  return true;
}
  


void
PacketRoutingEnv::NotifyPktRcv(Ptr<PacketRoutingEnv> entity, Ptr<NetDevice> netDev, NetDeviceContainer* nd, Ptr<const Packet> packet)
{  
  
  // define is train step flag
  entity->is_trainStep_flag = 0;

  //define if the packet is for signaling
  entity->m_signaling = 0;

  // define if the paccket is for OSPF (Not used)
  entity->m_ospfSignaling = false;


  //define 2-layer Header
  PppHeader ppp_head;
  
  //Define the packet
  Ptr<Packet> p;
  p = packet->Copy();
  
  //OSPF Analysis (Not used)
  
  //OSPFTag tagOspf;
  //p->PeekPacketTag(tagOspf);
  //
  //if(tagOspf.getType()==1){
  //  entity->m_signaling = 1;
  //  NS_LOG_UNCOND("HELLO MESSAGE");
  //}
  //if(tagOspf.getType()==2){
  //  if(!entity->m_lsaSeen[tagOspf.getLSANode()]){
  //    entity->m_ospfSignaling = true;
  //    entity->m_lsaSeen[tagOspf.getLSANode()] = true;
  //    entity->m_lsaTag = tagOspf;
  //  }
  //  entity->m_signaling = 1;
  //  
  //  //NS_LOG_UNCOND("LSA MESSAGE");
  //}

  //Remove Headers
  p->RemoveHeader(ppp_head);  
  p->RemoveHeader(entity->m_ipHeader);
  p->RemoveHeader(entity->m_udpHeader);
  entity->m_pckt = p->Copy();

  //Discarding if it is not UDP
  if(entity->m_ipHeader.GetProtocol()!=17){
    return ;
  }

  

  //Packet TAG
  MyTag tagCopy;
  p->PeekPacketTag(tagCopy);

  //Get Destination, source and last Hop
  entity->m_dest = tagCopy.GetFinalDestination();
  entity->m_src = entity->m_node->GetId();
  entity->m_lastHop = tagCopy.GetLastHop();
  
   //Broadcast Destination Address
  entity->m_destAddr = Mac48Address ("ff:ff:ff:ff:ff:ff");

  //Get Packet Size
  entity->m_size = p->GetSize();



  //Other Information
  entity->m_lengthType = ppp_head.GetProtocol();
  entity->m_recvDev = netDev;
  //Get Overlay Tunnel Index
  auto it = std::find(entity->m_overlayNeighbors.begin(), entity->m_overlayNeighbors.end(), entity->m_lastHop);
  if (it != entity->m_overlayNeighbors.end())
  {
      entity->m_overlayRecvIndex = distance(entity->m_overlayNeighbors.begin(), it);
  }
  else if(entity->m_lastHop !=1000 && tagCopy.GetSimpleValue()!=1)
  {
    NS_LOG_UNCOND(packet->ToString());
    NS_LOG_UNCOND("ERRRRRRRRR "<<entity->m_lastHop<<"    tag: "<<uint32_t(tagCopy.GetSimpleValue())<<"   UID: "<<packet->GetUid());
    NS_LOG_UNCOND("Node: "<<entity->m_node->GetId());
  }

  //Receiving Packets and treating them according to its type
  if(Simulator::Now().GetSeconds()>=(entity->m_cp+1.0)){
    entity->m_cp += 1.0;
    //NS_LOG_UNCOND("Time: "<<Simulator::Now().GetSeconds()<<"   "<<entity->m_node->GetId());
    for(size_t i =0; i<entity->m_overlayNeighbors.size();i++){
      uint64_t value;
      //if(entity->m_starting_overlay_packets[i].size()>=1){
      //  if(double(Simulator::Now().GetMilliSeconds() - entity->m_starting_overlay_packets[i][0].start_time)/2.0 >= entity->m_tunnelsDelay[i].back()){
      //    //NS_LOG_UNCOND("aqui ");
      //    if(entity->m_tunnelsDelay[i].size()>=entity->m_movingAverageObsSize){
      //      assert(!entity->m_tunnelsDelay[i].empty());
      //      entity->m_tunnelsDelay[i].erase(entity->m_tunnelsDelay[i].begin());
      //    }
      //    entity->m_tunnelsDelay[i].push_back(double(Simulator::Now().GetMilliSeconds() - entity->m_starting_overlay_packets[i][0].start_time)/2.0);
      //  }
      //  value = getAverage(entity->m_tunnelsDelay[i]); //std::max(getAverage(entity->m_tunnelsDelay[i])*2, double(Simulator::Now().GetMilliSeconds() - entity->m_starting_overlay_packets[i][0].start_time))/2.0;
      //  if(value>200 && entity->m_node->GetId()==0 && i==0){
      //    NS_LOG_UNCOND(getAverage(entity->m_tunnelsDelay[i]));
      //    if(getAverage(entity->m_tunnelsDelay[i])*2 > double(Simulator::Now().GetMilliSeconds() - entity->m_starting_overlay_packets[i][0].start_time)){
      //      entity->m_first_op_test += 1;
      //    } else{
      //      entity->m_second_op_test += 1;
      //  }
      //  }
      //  
      //} else {
      //  value = getAverage(entity->m_tunnelsDelay[i]);
      //  entity->m_first_op_test += 1;
      //}
      value = getAverage(entity->m_tunnelsDelay[i]);
      entity->m_tunnelsDelayGlobal[i].push_back(value);
      if(entity->m_node->GetId()==7){
        entity->m_bufferOccGlobal[0].push_back(entity->GetQueueLengthInBytes(entity->m_node_container->Get(3), 2));
        entity->m_bufferOccGlobal[1].push_back(entity->GetQueueLengthInBytes(entity->m_node_container->Get(7), 2));
      }
    }
  }
  // Type: 0 ----- Data Packets
  if(tagCopy.GetSimpleValue()==0x00){
    if(entity->m_lastHop==1000){
      m_packetsInjectedTotalGlobal += 1;
    }
    if(tagCopy.GetTrafficValable()==0){
      return ;
    }
    entity->m_packetStart = uint32_t(tagCopy.GetStartTime());
    if(entity->m_lastHop!=1000){
      if(entity->m_dest == entity->m_node->GetId()){
        //NS_LOG_UNCOND("Packet Delivered");
        m_packetsDeliveredGlobal += 1;
        m_end2endDelay.push_back(Simulator::Now().GetMilliSeconds()- entity->m_packetStart);
        m_cost.push_back((Simulator::Now().GetMilliSeconds()- entity->m_packetStart)*0.001);
        //NS_LOG_UNCOND("Times: "<<Simulator::Now().GetMilliSeconds()<<"   "<<entity->m_packetStart);
        //NS_LOG_UNCOND("Packets Delivered here "<<m_packetsDeliveredGlobal<<"    "<<m_end2endDelay.back()<<"   "<<getAverage(m_end2endDelay)<<"   "<<getAverage(m_cost));
      }
    }
    else{
      m_packetsInjectedGlobal += 1;
      m_bytesData += entity->m_size;
      //NS_LOG_UNCOND("Packets Injected here "<<m_packetsInjectedGlobal);
    }
  }

  if(tagCopy.GetSimpleValue()==4){
    entity->m_signaling=1;
    if(entity->m_tunnelsDelay[entity->m_overlayRecvIndex].size()>=entity->m_movingAverageObsSize){
      assert(!entity->m_tunnelsDelay[entity->m_overlayRecvIndex].empty());
      entity->m_tunnelsDelay[entity->m_overlayRecvIndex].erase(entity->m_tunnelsDelay[entity->m_overlayRecvIndex].begin());
    }
    entity->m_tunnelsDelay[entity->m_overlayRecvIndex].push_back(tagCopy.GetStartTime());
    entity->m_count_ping[entity->m_overlayRecvIndex] += 1;
    auto it = entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].begin();
    while(it->index != tagCopy.GetOverlayIndex()){
      it = entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].erase(it);
    }
    it = entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].erase(it);
    //for(size_t i =0;i<entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].size();i++){
    //  //if(entity->m_starting_overlay_packets[entity->m_overlayRecvIndex][i].index==tagCopy.GetOverlayIndex()){
    //  NS_LOG_UNCOND("HERE "<<i<<"   "<<entity->m_starting_overlay_packets[entity->m_overlayRecvIndex][i].index);
    //    //entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].erase(entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].begin());
    //    //break;
    //  //}
    //  //entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].erase(entity->m_starting_overlay_packets[entity->m_overlayRecvIndex].begin());
    //}
  }

  
  // Type: 3 ----- Overlay Signaling Packets
  if(tagCopy.GetSimpleValue()==3){
    entity->m_signaling=1;
    entity->m_timeStartOverlay = tagCopy.GetStartTime();
    entity->m_recvOverlayIndex = tagCopy.GetOverlayIndex();
    entity->sendOverlaySignalingUpdate(uint8_t(4));
    //entity->m_tunnelsDelay[entity->m_overlayRecvIndex] = Simulator::Now().GetMilliSeconds() - tagCopy.GetStartTime(); 
  }

  // Type 2: ---- Small Signaling Packets
  if(tagCopy.GetSimpleValue()==uint8_t(0x02)){
    entity->m_signaling=1;
    entity->m_pcktIdSign = tagCopy.GetIdValue();
    auto it = entity->m_packetsSent[entity->m_overlayRecvIndex].begin();
    while (it->uid != entity->m_pcktIdSign && it != entity->m_packetsSent[entity->m_overlayRecvIndex].end())
    {
      it++;
    }
    if(it != entity->m_packetsSent[entity->m_overlayRecvIndex].end()) {
      //NS_LOG_UNCOND("DATA PACKET "<<entity->m_pcktIdSign<<" ACKED");
      it=entity->m_packetsSent[entity->m_overlayRecvIndex].erase(it);
    }
    else{
      NS_LOG_UNCOND("ERR "<<entity->m_pcktIdSign<<"   Node: "<<entity->m_node->GetId()<<"   Time: "<<Simulator::Now().GetMilliSeconds()<<"   Id: "<<p->GetUid());
    }
    //NS_LOG_UNCOND("here");
  }

  // Type 1: ----- Big Signaling Packets
  if(tagCopy.GetSimpleValue()==0x01){
    entity->m_signaling=1;
    entity->m_NNIndex = tagCopy.GetNNIndex();
    entity->m_segIndex = tagCopy.GetSegIndex();
    entity->m_nodeIdSign = tagCopy.GetNodeId();
    //NS_LOG_UNCOND(entity->m_NNIndex<<"   "<<entity->m_segIndex);
  }

  //Discarding if it is Big Signaling in source
  if(entity->m_signaling && entity->m_node->GetId() != entity->m_dest){
    return ;
  }

  

  if(entity->m_signaling){
    if(tagCopy.GetSimpleValue()==1) m_bytesBigSignalling += entity->m_size;
    if(tagCopy.GetSimpleValue()==2) m_bytesSmallSignalling += entity->m_size;
    if(tagCopy.GetSimpleValue()==3) m_bytesOverlaySignalingForward += entity->m_size;
    if(tagCopy.GetSimpleValue()==4) m_bytesOverlaySignalingBack += entity->m_size;
  }
  
  //Printing INFO
  if(false){
    NS_LOG_UNCOND("..............................................................");
    NS_LOG_UNCOND("SimTime: "<<Simulator::Now().GetMilliSeconds());
    NS_LOG_UNCOND("Uid: "<<p->GetUid());
    NS_LOG_UNCOND("Node: "<<entity->m_node->GetId()<<"    ND: "<<netDev->GetIfIndex());
    NS_LOG_UNCOND("Destination: "<< entity->m_dest);
    NS_LOG_UNCOND("Last Hop: "<<entity->m_lastHop);
    NS_LOG_UNCOND("TAG: "<<uint32_t(tagCopy.GetSimpleValue()));
    NS_LOG_UNCOND(p->ToString());
    if(tagCopy.GetSimpleValue()==1) NS_LOG_UNCOND("BIG SIGNALING");
  }

  if(tagCopy.GetSimpleValue()==3 || tagCopy.GetSimpleValue()==4){
    return ;
  }
  


  
  
  // Get Start Time
  //if(tagCopy.GetSimpleValue()==0){
  //  
  //}
  

  
  

 

 
  
  //Notify
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
void
PacketRoutingEnv::simulationEnd(bool underlayTraff, double load)
{
  std::string filename;
  NS_LOG_UNCOND("Finishing Simulation");
  NS_LOG_UNCOND("Node: "<<m_node->GetId()<<"    First: "<<m_first_op_test<<"   Second: "<<m_second_op_test);
  for(size_t i=0;i<m_overlayNeighbors.size();i++){
    filename = "output_ping_"+std::to_string(underlayTraff)+"_" + std::to_string(m_node->GetId())+"_"+std::to_string(i)+".txt";
    NS_LOG_UNCOND(filename);
    NS_LOG_UNCOND(m_count_ping[i]);
    ofstream outputFile(filename);
    if(outputFile.is_open()){
      for(size_t j = 0;j<m_tunnelsDelayGlobal[i].size();j++){
        outputFile << m_tunnelsDelayGlobal[i][j] << std::endl;
      }
      outputFile.close();
    }
  }

  filename = "output_pingDiffs_"+std::to_string(underlayTraff)+"_"+std::to_string(m_node->GetId())+"_"+std::to_string(int(load*100))+".txt";
  ofstream outputFile1(filename);
  if(outputFile1.is_open()){
    for(size_t j = 0;j<m_pingDiffs.size();j++){
      outputFile1 << m_pingDiffs[j] << std::endl;
    }
    outputFile1.close();
  }
}
}// ns3 namespace