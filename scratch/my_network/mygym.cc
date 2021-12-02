/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
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
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 */

#include "mygym.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
//#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>
#include <vector>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv);

MyGymEnv::MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
  
  //Ptr<PointToPointNetDevice> m_array_p2pNetDevs[m_num_nodes][m_num_nodes];
  
  //std::vector<std::vector<int>> matrix(RR, vector<int>(CC);

  m_node = 0;
  m_lastEvNumPktsInQueue = 0;
  m_lastEvNode = 0;
  m_lastEvDev_idx = 1;
  m_fwdDev_idx = 1;
    //m_rxPktNum = 0;
}
  
MyGymEnv::MyGymEnv (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);
  //NetDeviceContainer m_list_p2pNetDevs = list_p2pNetDevs;
  
  m_node = node;
  m_lastEvNumPktsInQueue = 0;
  m_lastEvNode = 0;
  m_lastEvDev_idx = 1;
  m_fwdDev_idx = 1;
  //m_rxPktNum = 0;
}

MyGymEnv::MyGymEnv (Time stepTime, Ptr<Node> node)
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

  Simulator::Schedule (Seconds(0.0), &MyGymEnv::ScheduleNextStateRead, this);
}

void
MyGymEnv::ScheduleNextStateRead ()
{
  NS_LOG_FUNCTION (this);
  Simulator::Schedule (m_interval, &MyGymEnv::ScheduleNextStateRead, this);
  //Notify();
}

MyGymEnv::~MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
MyGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("MyGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<MyGymEnv> ()
  ;
  return tid;
}

void
MyGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<OpenGymSpace>
MyGymEnv::GetActionSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t num_devs = m_node->GetNDevices();
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (num_devs-1); // first dev is not p2p
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetActionSpace: " << space);
  return space;
  //uint32_t nodeNum = NodeList::GetNNodes ();
  //float low = 0.0;
  //float high = 100.0;
  //std::vector<uint32_t> shape = {nodeNum,};
  //std::string dtype = TypeNameGet<uint32_t> ();
  //Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
}

Ptr<OpenGymSpace>
MyGymEnv::GetObservationSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t num_devs = m_node->GetNDevices();
  float low = 0.0;
  float high = 100.0; // max buffer size --> to change depending on actual value (access to defaul sim param)
  std::vector<uint32_t> shape = {num_devs-1,}; // first dev is not p2p
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", GetObservationSpace: " << space);
  return space;
  }

bool
MyGymEnv::GetGameOver()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;
  //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetGameOver: " << isGameOver);
  return isGameOver;
}

uint32_t
MyGymEnv::GetQueueLength(Ptr<Node> node, uint32_t netDev_idx)
{
  Ptr<NetDevice> netDev = node->GetDevice (netDev_idx);
  //NS_LOG_UNCOND ("IsPointToPoint? : " << netDev->IsPointToPoint () << "");
  Ptr<PointToPointNetDevice> ptp_netDev = DynamicCast<PointToPointNetDevice> (netDev);
  //Ptr<PointToPointNetDevice> ptp_netDev = netDev->GetObject<PointToPointNetDevice> ();
  Ptr<Queue<Packet> > queue = ptp_netDev->GetQueue ();
  uint32_t backlog = (int) queue->GetNPackets();
  return backlog;
}

Ptr<OpenGymDataContainer>
MyGymEnv::GetObservation()
{
  NS_LOG_FUNCTION (this);
  printf("aqui obs\n");
  uint32_t num_devs = m_node->GetNDevices();
  std::vector<uint32_t> shape = {2};//{(num_devs-1)*50,};
  Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);
  for (uint32_t i=0 ; i<num_devs; i++){
    Ptr<NetDevice> netDev = m_node->GetDevice (i);
    //NS_LOG_UNCOND ("IsPointToPoint? : " << netDev->IsPointToPoint () << "");
    if (netDev->IsPointToPoint ()){ // Only P2P devices are considered: the first one is not a P2P
      uint32_t value = GetQueueLength (m_node, i);
      box->AddValue(value);
    }
  }

  NS_LOG_UNCOND ( "Node: " << m_node->GetId() << ", MyGetObservation: " << box);
  return box;
}

float
MyGymEnv::GetReward()
{
  NS_LOG_FUNCTION (this);
  //NS_LOG_UNCOND ("m_fwdDev_idx: " << m_fwdDev_idx);
  uint32_t value = GetQueueLength (m_node, m_fwdDev_idx);
  float reward = (float) value;
  //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetReward: " << reward);
  //NS_LOG_UNCOND ("Reward: Node with ID " << m_node->GetId() << ", net device with index " << m_fwdDev_idx << ", IF idx "<< (m_node->GetDevice(m_fwdDev_idx))->GetIfIndex() << ": New  queue size: " << reward << " packets");
  return reward;
}

std::string
MyGymEnv::GetExtraInfo()
{
  NS_LOG_FUNCTION (this);
  std::string myInfo = "currentNodeId=";
  myInfo += std::to_string(m_node->GetId());
  myInfo += ", currentNetDevIdx";
  myInfo += "=";
  if (m_lastEvDev_idx) {
    myInfo += std::to_string(m_lastEvDev_idx);
  }
  myInfo += ", currentFwdDevIdx";
  myInfo += "=";
  if (m_fwdDev_idx) {
    myInfo += std::to_string(m_fwdDev_idx);
  }
  NS_LOG_UNCOND("Node: " << m_node->GetId() << ", MyGetExtraInfo: " << myInfo);
  return myInfo;
}

bool
MyGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_UNCOND ("MyExecuteActions: " << action);
  Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  NS_LOG_UNCOND ("MyExecuteActionsDiscrete: " << discrete);
  uint32_t fwdDev = discrete->GetValue();
  m_fwdDev_idx = fwdDev+1; // since Dev_idx = 0 is not a p2p net dev, we have to add one
  
  NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyExecuteActions: " << m_fwdDev_idx);
  return true;
}
  

void
MyGymEnv::CountPktInQueueEvent(Ptr<MyGymEnv> entity, Ptr<PointToPointNetDevice> ptpnd, uint32_t oldValue, uint32_t newValue)
  {
    entity->m_lastEvNode = (ptpnd->GetNode())->GetId();
    entity->m_lastEvDev_idx = ptpnd->GetIfIndex();
    entity->m_lastEvNumPktsInQueue = newValue;
    
    Ptr< Channel > p2p_channel = ptpnd->GetChannel();
    Ptr<NetDevice> dev_a = p2p_channel->GetDevice(0);
    Ptr<NetDevice> dev_b = p2p_channel->GetDevice(1);
  
    //NS_LOG_UNCOND ("Event (Node = " << (dev_a->GetNode())->GetId() << " -> Node = " << (dev_b->GetNode())->GetId() << ") ");
    //NS_LOG_UNCOND ("Event (Node = " << entity->m_lastEvNode << ", IF idx = "<< entity->m_lastEvDev_idx << ")--> New queue size: " << newValue << " packets");
  }
  
  void
  MyGymEnv::NotifyPktInQueueEvent(Ptr<MyGymEnv> entity, Ptr<PointToPointNetDevice> ptpnd, uint32_t oldValue, uint32_t newValue)
  {
    entity->m_lastEvNode = (ptpnd->GetNode())->GetId();
    //ptpnd->SetIfIndex(1);
    entity->m_lastEvDev_idx = ptpnd->GetIfIndex();
    entity->m_lastEvNumPktsInQueue = newValue;
    
    Ptr< Channel > p2p_channel = ptpnd->GetChannel();
    Ptr<NetDevice> dev_a = p2p_channel->GetDevice(0);
    Ptr<NetDevice> dev_b = p2p_channel->GetDevice(1);
    
    NS_LOG_UNCOND ("Event (Node = " << (dev_a->GetNode())->GetId() << " -> Node = " << (dev_b->GetNode())->GetId() << ") ");
    NS_LOG_UNCOND ("Event (Node = " << entity->m_lastEvNode << ", IF idx = "<< entity->m_lastEvDev_idx << ")--> New queue size: " << newValue << " packets");
    entity->Notify();
  }
  
  void
  MyGymEnv::CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, const Address & srcAddr)
  {
    //m_rxPkts[sinkId]++;
    //NS_LOG_UNCOND("EVENT == "<<sinkId<<"   "<<srcAddr<<"                      ");
    Ptr<Packet> copy = packet->Copy ();
    Ipv4Header iph;
    copy->RemoveHeader (iph);
    NS_LOG_UNCOND("HEADER    "<<iph);
    packet->PrintByteTags(std::cout);
  }

  

} // ns3 namespace