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

#ifndef PING_BACK_PACKET_MANAGER_H
#define PING_BACK_PACKET_MANAGER_H
#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "packet-manager.h"


using namespace std;
namespace ns3 {

class Node;

class PingBackPacketManager : public PacketManager
{
public:
  PingBackPacketManager();
  PingBackPacketManager(Ptr<Node> node, vector<int> neighbors);

  bool receivePacket(Ptr<Packet> packet, Ptr<NetDevice> receivingNetDev);
  vector<float> m_tunnelsDelay [MAX_TUNNELS];
  vector<SentPacket> m_sentPingForwardPackets[MAX_TUNNELS];
  void addSentPingForwardPacket(uint64_t id, uint64_t start_time);
  float getMaxTimePingForwardPacketSent(uint32_t index);
  string getInfo();
  void losePacket();
  string getLostPackets();
  void setMovingAverageSize(uint32_t value);

private:
  uint32_t m_overlayTunnelIndex;
  uint32_t m_pingPacketIndex;
  uint32_t m_movingAverageSize=20;
  Ptr<NetDevice> m_receivingNetDev;
};

}
#endif // PACKET_ROUTING_GYM_ENTITY_H
