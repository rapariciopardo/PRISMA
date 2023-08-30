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

NS_LOG_COMPONENT_DEFINE ("PacketRoutingEnv");

NS_OBJECT_ENSURE_REGISTERED (PacketRoutingEnv);

PacketRoutingEnv::PacketRoutingEnv ()
{
  NS_LOG_FUNCTION (this);
}
  
PacketRoutingEnv::PacketRoutingEnv (Ptr<Node> node, NodeContainer nodes, uint64_t linkRateValue, bool activateSignaling, double signPacketSize, vector<int> overlayNeighbors, int *nodes_starting_address)
{
  NS_LOG_FUNCTION (this);
  
  m_bigSignalingPacketManager = new BigSignalingPacketManager(node, overlayNeighbors);
  m_smallSignalingPacketManager = new SmallSignalingPacketManager(node, overlayNeighbors);
  m_pingForwardPacketManager = new PingForwardPacketManager(node, overlayNeighbors);
  m_pingBackPacketmanager = new PingBackPacketManager(node, overlayNeighbors);
  m_dataPacketManager = new DataPacketManager(node, overlayNeighbors, nodes_starting_address, nodes);
  m_dataPacketManager->setSmallSignalingPacketSize(signPacketSize);
  m_dataPacketManager->setPingBackPacketManager(m_pingBackPacketmanager);
}

void 
PacketRoutingEnv::configDataPacketManager(bool obs_bufferLength){
  m_dataPacketManager->setObsBufferLength(obs_bufferLength);
}

void
PacketRoutingEnv::configPingBackPacketManager(uint32_t movingAverageSize){
  m_pingBackPacketmanager->setMovingAverageSize(movingAverageSize);
}

void
PacketRoutingEnv::setNetDevicesContainer(NetDeviceContainer* nd){
  m_dataPacketManager->setNetDevContainer(*nd);
}

void
PacketRoutingEnv::setTunnelsMaxDelays(vector<vector<double>> tunnelMaxDelays){
  m_pingBackPacketmanager->setTunnelsMaxDelays(tunnelMaxDelays);
}

void
PacketRoutingEnv::setTrainConfig(bool train){
  m_train = train;
}

void 
PacketRoutingEnv::setPingPacketIntervalTime(float pingPacketIntervalTime){
  m_dataPacketManager->setPingPacketIntervalTime(pingPacketIntervalTime);
}

void
PacketRoutingEnv::ScheduleNextStateRead ()
{
  NS_LOG_FUNCTION (this);
  Simulator::Schedule (m_interval, &PacketRoutingEnv::ScheduleNextStateRead, this);
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
  return m_dataPacketManager->getActionSpace();
}

Ptr<OpenGymSpace>
PacketRoutingEnv::GetObservationSpace()
{
  return m_dataPacketManager->getObservationSpace();
}

bool
PacketRoutingEnv::GetGameOver()
{
  if(is_trainStep_flag==0) return m_dataPacketManager->getGameOver();
  else return false;
}

Ptr<OpenGymDataContainer>
PacketRoutingEnv::GetObservation()
{
  Ptr<OpenGymBoxContainer<int32_t> > box = CreateObject<OpenGymBoxContainer<int32_t> >(m_dataPacketManager->getObsShape());
  if(is_trainStep_flag==0) {
    if (m_packetType==DATA_PACKET) return m_dataPacketManager->getObservation();
    else box->AddValue(1000);
  }
  else {
    box->AddValue(-1);
  }
  return  box;
}

float
PacketRoutingEnv::GetReward()
{
  if(is_trainStep_flag==0) return m_dataPacketManager->getReward();
  else return -1;
}

std::string
PacketRoutingEnv::GetExtraInfo()
{
  std::string myInfo;
  if (is_trainStep_flag==1){
    // notify the end of the episode to collect stats
    return m_dataPacketManager->getInfo();
  }
  else{
    myInfo="-2";

  }
  
  if(m_packetType==BIG_SIGN_PACKET){
    myInfo = m_bigSignalingPacketManager->getInfo();
  }
  
  if(m_packetType==DATA_PACKET){
    if(is_trainStep_flag==0) myInfo = m_dataPacketManager->getInfo();
    else{
      std::string invalidRet("-1,");
      myInfo = invalidRet;
    }
  }

  if(m_packetType==SMALL_SIGN_PACKET){
    myInfo = m_smallSignalingPacketManager->getInfo();
  }

  return myInfo;
}

bool
PacketRoutingEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  bool sent = true;
  if(m_packetType==DATA_PACKET){
    if (is_trainStep_flag==1){
      return true;
    } else{
      if(m_train){
        //send small signaling
        m_dataPacketManager->sendSmallSignalingPacket();
     }
     //Send data packet
      return m_dataPacketManager->sendPacket(action);
    }
  }
  return sent;
}
 
void
PacketRoutingEnv::initialize()
{
  is_trainStep_flag = 1;
  Notify();
  
}

void
PacketRoutingEnv::mapOverlayNodes(std::vector <int> map_overlay_array)
{
  m_dataPacketManager->m_map_overlay_array = map_overlay_array;
  m_bigSignalingPacketManager->m_map_overlay_array = map_overlay_array;
  m_pingBackPacketmanager->m_map_overlay_array = map_overlay_array;
}

void
PacketRoutingEnv::NotifyPktRcv(Ptr<PacketRoutingEnv> entity, Ptr<NetDevice> netDev, NetDeviceContainer* nd, Ptr<const Packet> packet)
{  
  //Redefine is train step flag
  entity->is_trainStep_flag = 0;
  
  //Define the packet
  Ptr<Packet> p;
  p = packet->Copy();
  
  //Define Tag
  MyTag tagCopy;
  p->PeekPacketTag(tagCopy);

  if(tagCopy.GetSimpleValue()==0 && tagCopy.GetSource()==4 && tagCopy.GetFinalDestination()==3 && packet->GetUid()==12425)
  NS_LOG_UNCOND("PacketRoutingEnv::NotifyPktRcv: packet->GetUid()="<<packet->GetUid()<<", tagCopy.GetSimpleValue()="<<tagCopy.GetSimpleValue()<<", tagCopy.GetSource()="<<tagCopy.GetSource()<<", tagCopy.GetFinalDestination()="<<tagCopy.GetFinalDestination()<<", tagCopy.GetStartTime()="<<tagCopy.GetStartTime()<<", tagCopy.GetLastHop()="<<tagCopy.GetLastHop()<<", tagCopy.GetNextHop()="<<tagCopy.GetNextHop() << ", node id="<<entity->m_dataPacketManager->m_node->GetId()<<", packet size="<<packet->GetSize()<<", packet type="<<PacketType(tagCopy.GetSimpleValue())<<", p");

  //Get packet type
  entity->m_packetType = PacketType(tagCopy.GetSimpleValue());
  
  bool valid = true;
  if(entity->m_packetType==DATA_PACKET){
    valid = entity->m_dataPacketManager->receivePacket(p, netDev);
  } else if(entity->m_packetType==BIG_SIGN_PACKET){
    valid = entity->m_bigSignalingPacketManager->receivePacket(p);
  } else if(entity->m_packetType==SMALL_SIGN_PACKET){
    valid = entity->m_smallSignalingPacketManager->receivePacket(p);
  } else if(entity->m_packetType==PING_FORWARD_PACKET){
    entity->m_pingForwardPacketManager->receivePacket(p, netDev);
    return ;
  } else if(entity->m_packetType==PING_BACK_PACKET){
    entity->m_pingBackPacketmanager->receivePacket(p, netDev);
    return ;
  }

  //If packet is not valid, don't notify
  if(!valid) return ;

  //Notify
  entity->Notify();
}

}// ns3 namespace