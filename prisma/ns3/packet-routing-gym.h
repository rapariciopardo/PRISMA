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
#include <vector>

#define MAX_TUNNELS 11

namespace ns3 {

class Node;
class NetDevice;
class PointToPointNetDevice;
class NetDeviceContainer;

struct StartingOverlayPacket{
  uint32_t index;
  uint64_t start_time;
};

struct StartingDataPacket{
  uint64_t uid;
  uint64_t start_time;
};

//class Packet;
//class QueueBase;


class PacketRoutingEnv : public OpenGymEnv
{
public:
  PacketRoutingEnv ();
  PacketRoutingEnv (Ptr<Node> node, uint32_t numberOfNodes, uint64_t linkRateValue, bool activateSignaling, double signPacketSize, vector<int> overlayNeighbors);
  PacketRoutingEnv (Time stepTime, Ptr<Node> node);
  void setOverlayConfig(vector<int> overlayNeighbors, bool activateOverlaySignaling, uint32_t nPacketsOverlaySignaling, uint32_t movingAverageObsSize, vector<int> map_overlay_array);
  void setNetDevicesContainer(NetDeviceContainer* nd);
  void setTrainConfig(bool train);
  void setLossPenalty(double lossPenalty);
  void setPingAsObs(bool pingAsObs);
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
  static std::vector<uint32_t> m_rxPkts;

  // the function has to be static to work with MakeBoundCallback
  // that is why we pass pointer to PacketRoutingEnv instance to be able to store the context (node, etc)
 
  static void NotifyPktRcv(Ptr<PacketRoutingEnv> entity, Ptr<NetDevice> netDev, NetDeviceContainer* nd, Ptr<const Packet> packet);
  static void NotifyTrainStep(Ptr<PacketRoutingEnv> entity);
  bool is_trainStep_flag;
  NodeContainer* m_node_container;
  void simulationEnd(bool underlayTraff, double load);
  void setPingTimeout(uint32_t maxBufferSize, uint32_t linkCapacity, uint32_t propagationDelay);


private:
  void ScheduleNextStateRead();
  uint32_t GetQueueLength(Ptr<Node> node, uint32_t netDev_idx);
  uint32_t GetQueueLengthInBytes(Ptr<Node> node, uint32_t netDev_idx);
  uint32_t getNbPacketsBuffered();
  std::string GetLostPackets();
  uint32_t mapOverlayNode(uint32_t underlayNode);
  void sendOverlaySignalingUpdate(uint8_t type);

  //bool SetCw(Ptr<Node> node, uint32_t cwMinValue=0, uint32_t cwMaxValue=0);

  Time m_interval = Seconds(0.1);
  //uint32_t m_num_nodes;
  Ptr<Node> m_node;
  uint32_t m_dest;
  uint32_t m_src;
  uint32_t m_lastHop;
  vector<int> m_overlayNeighbors;
  static uint32_t m_n_nodes;
  Address m_srcAddr;
  Address m_destAddr;
  UdpHeader m_udpHeader;
  Ipv4Header m_ipHeader;

  Ptr<Packet> m_pckt;
  uint16_t m_lengthType = 2054;
  uint32_t m_size;
  float m_packetRate = 500.0;
  uint32_t m_packetStart;
  bool m_isGameOver;
  std::vector<uint32_t> m_obs_shape;
  Ptr<NetDevice> m_recvDev;
  vector<bool> m_lsaSeen;
  
  
  OSPFTag m_lsaTag;
  bool m_ospfSignaling;

  uint32_t m_fwdDev_idx;  // Last net device selected to forward the packet (last action)
  uint32_t m_fwdDev_idx_overlay;
  uint32_t m_lastEvDev_idx;  // Last net device triggering an event 
  uint32_t m_lastEvNode;  // Node where last net device triggering an event 
  uint32_t m_lastEvNumPktsInQueue; // Queue backlog of last event device

  bool m_activateSignaling;
  double m_signPacketSize; 
  uint64_t m_pcktIdSign;
  uint32_t m_nodeIdSign;
  uint32_t m_NNIndex;
  uint32_t m_segIndex;
  int m_signaling;

  
  bool m_activateOverlaySignaling;
  bool m_train; 
  uint32_t m_countSendPackets;
  vector<uint64_t> m_tunnelsDelay [MAX_TUNNELS];
  vector<StartingOverlayPacket> m_starting_overlay_packets [MAX_TUNNELS];
  vector<uint64_t> m_tunnelsDelayGlobal [MAX_TUNNELS];
  vector<uint32_t> m_bufferOccGlobal[MAX_TUNNELS]; 
  int m_count_ping [MAX_TUNNELS];
  int m_overlayIndex [MAX_TUNNELS];
  int m_overlayRecvIndex;
  uint32_t m_nPacketsOverlaySignaling;
  uint32_t m_movingAverageObsSize;

  uint32_t m_packetsDropped;
  uint32_t m_packetsDelivered;
  static int m_packetsDeliveredGlobal;
  static int m_packetsInjectedGlobal;
  static int m_packetsDroppedGlobal;
  static int m_packetsDroppedTotalGlobal;
  static int m_packetsInjectedTotalGlobal;
  static int m_testPacketsDroppedGlobal;
  static int m_bytesData;
  static int m_bytesBigSignalling;
  static int m_bytesSmallSignalling;
  static int m_bytesOverlaySignalingForward;
  static int m_bytesOverlaySignalingBack;
  static std::vector<int> m_end2endDelay;
  static std::vector<float> m_cost;

  double m_loss_penalty;

  NetDeviceContainer m_all_nds;

  uint64_t m_timeStartOverlay;
  uint32_t m_recvOverlayIndex;

  vector<StartingDataPacket> m_packetsSent [MAX_TUNNELS];
  std::string m_lost_packets;

  float m_cp = 0.0; 
  int m_first_op_test = 0; 
  int m_second_op_test = 0;

  float m_pingTimeout[MAX_TUNNELS];
  float m_lastPingOut;
  vector<float> m_pingDiffs;

  vector<int> m_map_overlay_array;
  bool m_first[MAX_TUNNELS]={false};

  bool m_pingAsObs = true;


 


  
  //uint64_t m_rxPktNum;

};

}


#endif // PACKET_ROUTING_GYM_ENTITY_H
