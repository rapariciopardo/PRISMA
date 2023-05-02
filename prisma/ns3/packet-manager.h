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

#ifndef PACKET_MANAGER_H
#define PACKET_MANAGER_H

#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/ppp-header.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "enum-and-constants.h"
#include "compute-stats-v2.h"



#define MAX_TUNNELS 11
#define MAX_NODES 11

using namespace std;

namespace ns3 {
struct SentPacket{
  uint64_t uid;
  PacketType type;
  uint64_t start_time;
};
class Node;




class PacketManager
{
public:
  PacketManager();
  PacketManager (Ptr<Node> node, vector<int> neighbors);
  

  void step(Ptr<Packet> packet);
  void receivePacket(Ptr<Packet> packet);
  std::string getInfo();
  void writeStats();
  std::string getLostPackets();
  void setNetDevContainer(NetDeviceContainer netDevs);


  Ptr<Node> m_node;
  vector<int> m_neighbors;
  ComputeStats *m_computeStats;

  Ptr<Packet> m_packet;
  PacketType m_packetType;
  uint32_t m_lastHop;
  uint32_t m_source;
  uint32_t m_nextHop;
  uint32_t m_destination;
  uint64_t m_sourceTimeStamp;
  uint32_t m_packetSize;
  uint64_t m_packetUid;
  bool m_arrivedAtOrigin;
  bool m_arrivedAtFinalDest;

  Ipv4Header m_packetIpHeader;
  UdpHeader m_packetUdpHeader;
  PppHeader m_packetPppHeader;

  std::string m_nextHopIp;
  std::vector<SentPacket> m_sentPackets;
  std::vector<SentPacket> m_lostPackets;

  Address m_destAddr = Mac48Address ("ff:ff:ff:ff:ff:ff");
  NetDeviceContainer m_netDevs;



private:
  
};

}


#endif // PACKET_ROUTING_GYM_ENTITY_H
