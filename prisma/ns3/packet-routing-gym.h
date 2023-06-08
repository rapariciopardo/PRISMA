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
 * 
 */

#ifndef PACKET_ROUTING_GYM_ENTITY_H
#define PACKET_ROUTING_GYM_ENTITY_H

#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "point-to-point-net-device.h"
#include "ospf-tag.h"
#include "enum-and-constants.h"
#include "data-packet-manager.h"
#include "big-signaling-packet-manager.h"
#include "small-signaling-packet-manager.h"
#include "ping-forward-packet-manager.h"
#include "ping-back-packet-manager.h"
#include <vector>

namespace ns3 {

class Node;
class NetDevice;
class PointToPointNetDevice;
class NetDeviceContainer;



class PacketRoutingEnv : public OpenGymEnv
{
public:
  PacketRoutingEnv ();
  PacketRoutingEnv (Ptr<Node> node, NodeContainer nodes, uint64_t linkRateValue, bool activateSignaling, double signPacketSize, vector<int> overlayNeighbors, int *nodes_starting_address);
  void ScheduleNextStateRead ();
  void setNetDevicesContainer(NetDeviceContainer* nd);
  void setTrainConfig(bool train);
  void setPingPacketIntervalTime(float pingPacketIntervalTime);
  void configDataPacketManager(bool obs_bufferLength);
  void configPingBackPacketManager(uint32_t movingAverageSize);
  void mapOverlayNodes(std::vector <int> map_overlay_array);
  
  virtual ~PacketRoutingEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  Ptr<OpenGymSpace> GetActionSpace();
  Ptr<OpenGymSpace> GetObservationSpace();
  bool GetGameOver();
  Ptr<OpenGymDataContainer> GetObservation();
  float GetReward();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);
  static void dropPacket(Ptr<PacketRoutingEnv> entity, Ptr<const Packet> packet);
  void initialize();
  static std::vector<uint32_t> m_rxPkts;
  ns3::NodeContainer m_nodes;  
  // the function has to be static to work with MakeBoundCallback
  // that is why we pass pointer to PacketRoutingEnv instance to be able to store the context (node, etc)
 
  static void NotifyPktRcv(Ptr<PacketRoutingEnv> entity, Ptr<NetDevice> netDev, NetDeviceContainer* nd, Ptr<const Packet> packet);
  static void NotifyTrainStep(Ptr<PacketRoutingEnv> entity);
  bool is_trainStep_flag;
  
private:
  Time m_interval = Seconds(0.1);
  DataPacketManager *m_dataPacketManager;
  BigSignalingPacketManager *m_bigSignalingPacketManager;
  SmallSignalingPacketManager *m_smallSignalingPacketManager;
  PingForwardPacketManager *m_pingForwardPacketManager;
  PingBackPacketManager *m_pingBackPacketmanager;
  PacketType m_packetType;
  
  bool m_train; 
};

}


#endif // PACKET_ROUTING_GYM_ENTITY_H
