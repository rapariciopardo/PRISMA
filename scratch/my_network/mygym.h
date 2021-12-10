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

#ifndef MY_GYM_ENTITY_H
#define MY_GYM_ENTITY_H

#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include <vector>

namespace ns3 {

class Node;
class NetDevice;
class PointToPointNetDevice;
class NetDeviceContainer;

//class Packet;
//class QueueBase;


class MyGymEnv : public OpenGymEnv
{
public:
  MyGymEnv ();
  MyGymEnv (Ptr<Node> node, uint32_t numberOfNodes, uint64_t linkRateValue);
  MyGymEnv (Time stepTime, Ptr<Node> node);
  virtual ~MyGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  Ptr<OpenGymSpace> GetActionSpace();
  Ptr<OpenGymSpace> GetObservationSpace();
  bool GetGameOver();
  Ptr<OpenGymDataContainer> GetObservation();
  float GetReward();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);
  static std::vector<uint32_t> m_rxPkts;

  // the function has to be static to work with MakeBoundCallback
  // that is why we pass pointer to MyGymEnv instance to be able to store the context (node, etc)
  static void CountPktInQueueEvent(Ptr<MyGymEnv> entity, Ptr<PointToPointNetDevice> ptpnd, uint32_t oldValue, uint32_t newValue); // time-driven step
  static void NotifyPktInQueueEvent(Ptr<MyGymEnv> entity, Ptr<PointToPointNetDevice> ptpnd, uint32_t oldValue, uint32_t newValue); // event-driven step
  static void CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, Ptr<Ipv4> header, uint32_t something);
  static void NotifyPktRcv(Ptr<MyGymEnv> entity, Ptr<Node> node, NetDeviceContainer* nd, Ptr<const Packet> packet);
  static void NotifyPktRcvCSMA(Ptr<MyGymEnv> entity, Ptr<Node> node, NetDeviceContainer* nd, Ptr<const Packet> packet);


private:
  void ScheduleNextStateRead();
  uint32_t GetQueueLength(Ptr<Node> node, uint32_t netDev_idx);
  //bool SetCw(Ptr<Node> node, uint32_t cwMinValue=0, uint32_t cwMaxValue=0);

  Time m_interval = Seconds(0.1);
  //uint32_t m_num_nodes;
  Ptr<Node> m_node;
  uint32_t m_dest;
  static uint32_t m_n_nodes;
  Address m_srcAddr;
  Address m_destAddr;
  Ptr<Packet> m_pckt;
  uint16_t m_lengthType = 2054;
  uint32_t m_size;
  float m_packetRate = 500.0;
  bool m_isGameOver;
  //NetDeviceContainer m_list_p2pNetDevs;
  uint32_t m_fwdDev_idx;  // Last net device selected to forward the packet (last action)
  uint32_t m_lastEvDev_idx;  // Last net device triggering an event 
  uint32_t m_lastEvNode;  // Node where last net device triggering an event 
  uint32_t m_lastEvNumPktsInQueue; // Queue backlog of last event device

  
  //uint64_t m_rxPktNum;

};

}


#endif // MY_GYM_ENTITY_H
