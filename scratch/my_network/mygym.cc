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
#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"

//#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>
#include <vector>



namespace ns3 {
uint32_t MyGymEnv::m_n_nodes;
uint32_t MyGymEnv::m_counter;


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
  
MyGymEnv::MyGymEnv (Ptr<Node> node, uint32_t numberOfNodes, uint64_t packetRate)
{
  NS_LOG_FUNCTION (this);
  //NetDeviceContainer m_list_p2pNetDevs = list_p2pNetDevs;
  m_packetRate = packetRate;
  m_n_nodes = numberOfNodes;
  m_node = node;
  m_counter = 0;
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
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (num_devs); // first dev is not p2p
  NS_LOG_UNCOND ("Node: " << m_node->GetId()-(m_n_nodes-1) << ", GetActionSpace: " << space);
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
  std::vector<uint32_t> shape = {num_devs+1,}; // first dev is not p2p
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  NS_LOG_UNCOND ("Node: " << m_node->GetId()-(m_n_nodes-1) << ", GetObservationSpace: " << space);
  return space;
  }

bool
MyGymEnv::GetGameOver()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;
  NS_LOG_UNCOND(m_node->GetId()<<"     "<<m_dest);
  if (m_node->GetId() == m_dest) m_counter++;

  isGameOver = (m_counter==m_n_nodes);
  //NS_LOG_UNCOND ("Node: " << m_node->GetId() << ", MyGetGameOver: " << isGameOver);
  return isGameOver;
}

uint32_t
MyGymEnv::GetQueueLength(Ptr<Node> node, uint32_t netDev_idx)
{
  Ptr<NetDevice> netDev = node->GetDevice (netDev_idx);
  //NS_LOG_UNCOND ("IsPointToPoint? : " << netDev->IsPointToPoint () << "");
  Ptr<CsmaNetDevice> csma_netDev = DynamicCast<CsmaNetDevice> (netDev);
  //Ptr<PointToPointNetDevice> ptp_netDev = netDev->GetObject<PointToPointNetDevice> ();
  Ptr<Queue<Packet> > queue = csma_netDev->GetQueue ();
  uint32_t backlog = (int) queue->GetNPackets();
  return backlog;
}

Ptr<OpenGymDataContainer>
MyGymEnv::GetObservation()
{
  NS_LOG_FUNCTION (this);
  uint32_t num_devs = m_node->GetNDevices();
  NS_LOG_UNCOND("N devices: "<<num_devs);
  std::vector<uint32_t> shape = {num_devs+1};//{(num_devs-1)*50,};
  Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);
  box->AddValue(m_dest);
  for (uint32_t i=0 ; i<num_devs; i++){
    Ptr<NetDevice> netDev = m_node->GetDevice (i);
    //NS_LOG_UNCOND ("IsPointToPoint? : " << netDev->IsPointToPoint () << "");
    //if (netDev->IsPointToPoint ()){ // Only P2P devices are considered: the first one is not a P2P
    uint32_t value = GetQueueLength (m_node, i);
    box->AddValue(value);
    //}
  }

  NS_LOG_UNCOND ( "Node: " << m_node->GetId()-(m_n_nodes-1) << ", MyGetObservation: " << box);
  return box;
}

float
MyGymEnv::GetReward()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_UNCOND ("m_fwdDev_idx: " << m_fwdDev_idx);
  uint32_t value = GetQueueLength (m_node, m_fwdDev_idx);
  float transmission_time = m_size/(m_packetRate*(1/8));
  float reward = transmission_time + transmission_time*(float) value;
  NS_LOG_UNCOND ("Node: " << m_node->GetId()-(m_n_nodes-1) << ", MyGetReward: " << reward);
  //NS_LOG_UNCOND ("Reward: Node with ID " << m_node->GetId() << ", net device with index " << m_fwdDev_idx << ", IF idx "<< (m_node->GetDevice(m_fwdDev_idx))->GetIfIndex() << ": New  queue size: " << reward << " packets");
  return reward;
}

std::string
MyGymEnv::GetExtraInfo()
{
  //NS_LOG_FUNCTION (this);
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
  //NS_LOG_UNCOND("Node: " << m_node->GetId() << ", MyGetExtraInfo: " << myInfo);
  return myInfo;
}

bool
MyGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_UNCOND ("MyExecuteActions: " << action);
  Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  NS_LOG_UNCOND ("MyExecuteActionsDiscrete: " << discrete);
  uint32_t m_fwdDev_idx = discrete->GetValue();
  std::cout<<"New Path: ";
  std::cin>>m_fwdDev_idx;
  NS_LOG_UNCOND ("Node: " << m_node->GetId()-(m_n_nodes-1) << ", MyExecuteActions: " << m_fwdDev_idx);
  Ptr<CsmaNetDevice> dev = DynamicCast<CsmaNetDevice>(m_node->GetDevice(m_fwdDev_idx));
  NS_LOG_UNCOND(dev->GetAddress());
  NS_LOG_UNCOND(m_srcAddr<<"     "<<m_destAddr<< "    "<<m_lengthType);
  bool sent = dev->SendFrom(m_pckt, m_srcAddr, m_destAddr, m_lengthType);
  //Simulator::Stop (Seconds (10.0));
  //Simulator::Run ();

  NS_LOG_UNCOND(sent);

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
  MyGymEnv::CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, Ptr<Ipv4> header, uint32_t something)
  {
    //m_rxPkts[sinkId]++;
    //NS_LOG_UNCOND("EVENT == "<<sinkId<<"   "<<srcAddr<<"                      ");
    Ptr<Packet> copy = packet->Copy ();
    Ipv4Header iph;
    copy->RemoveHeader (iph);
    NS_LOG_UNCOND("HEADER    "<<iph);
    packet->PrintByteTags(std::cout);
  }

  void
  MyGymEnv::NotifyPktRcv(Ptr<MyGymEnv> entity, Ptr<Node> node, NetDeviceContainer* nd, Ptr<const Packet> packet)
  {
    //uint8_t buf_add[6];
    
    EthernetHeader head;
    ArpHeader iph;
    //NS_LOG_UNCOND(nd_sw->Get(idx)->GetAddress());
    
    Ptr<Packet> p = packet->Copy();
    entity->m_size = p->GetSize();
    NS_LOG_UNCOND("Node "<<node->GetId()-(m_n_nodes-1));

    //Remove Mac Header
    p->RemoveHeader(head);
    entity->m_pckt = p->Copy();
    entity->m_lengthType = head.GetLengthType();
    entity->m_srcAddr = head.GetSource();
    entity->m_destAddr = head.GetDestination();
    //entity->m_destAddr.CopyTo(buf_add);
    //entity->m_dest = (uint32_t)buf_add[5];
    //NS_LOG_UNCOND("Destination: "<<entity->m_destAddr);
        
    //head.Print(std::cout);
    
    
    
    //Peeking IP Header
    p->PeekHeader(iph);
    //p->Print(std::cout);
    NS_LOG_UNCOND("Src/dest Ip Addr "<<iph.GetSourceIpv4Address()<<"    "<<iph.GetDestinationIpv4Address());
    
    //entity->m_srcAddr = iph.GetSourceHardwareAddress();

    //head.Print(std::cout);
    for(uint32_t i = 0;i<nd->GetN();i++){
      Ptr<NetDevice> dev = nd->Get(i);
      Ptr<Node> n = dev->GetNode();
      Ptr<Ipv4> ipv4 = n->GetObject<Ipv4> ();
      Ipv4InterfaceAddress ipv4_int_addr = ipv4->GetAddress (1, 0);
      Ipv4Address ip_addr = ipv4_int_addr.GetLocal ();

      //NS_LOG_UNCOND(ip_addr);
      if(ip_addr == iph.GetDestinationIpv4Address()){
        //Address mac48_dest_addr = dev->GetAddress();
        
        //mac48_dest_addr.CopyTo(buf_add);
        entity->m_dest = dev->GetNode()->GetId()+5;//(uint32_t)buf_add[5];
        NS_LOG_UNCOND("Match dest"<<entity->m_dest);
        
        //for(int i=0;i<6;i++){
        //  NS_LOG_UNCOND((uint32_t) buf_add[i]);
        //}
        break;
      }
    }
    entity->Notify();
  }

} // ns3 namespace