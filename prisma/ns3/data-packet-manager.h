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

#ifndef DATA_PACKET_MANAGER_H
#define DATA_PACKET_MANAGER_H
#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "packet-manager.h"
#include "ping-back-packet-manager.h"


using namespace std;
namespace ns3 {

class Node;


class DataPacketManager : public PacketManager
{
public:
  DataPacketManager();
  DataPacketManager(Ptr<Node> node, vector<int> neighbors, int *nodes_starting_address, ns3::NodeContainer nodes_switch);
  void setSmallSignalingPacketSize(uint32_t signPacketSize);
  void setPingPacketIntervalTime(float pingBackIntervalTime);
  void setPingBackPacketManager(PingBackPacketManager *pingBackPacketManager);

  bool receivePacket(Ptr<Packet> packet, Ptr<NetDevice> receivingNetDev);

  string getInfo();
  static void dropPacket(DataPacketManager *entity, Ptr<const Packet> packet);
  string getLostPackets();
  uint32_t getQueueLengthInBytes(Ptr<Node> node, uint32_t netDev_idx);
  Ptr<OpenGymDataContainer> getObservation();
  float getReward();
  bool getGameOver();
  Ptr<OpenGymSpace> getObservationSpace();
  vector<uint32_t> getObsShape();
  Ptr<OpenGymSpace> getActionSpace();
  bool sendPacket(uint32_t action);
  void sendSmallSignalingPacket();
  void sendPingForwardPacket(uint32_t overlayIndex);
  void sendPingPackets();
  void setObsBufferLength(bool value);
  
private:
  Time m_pingPacketInterval = Seconds(1.0);
  NodeContainer m_nodes_switch;
  bool m_obs_bufferLength = false;
  PingBackPacketManager *m_pingBackPacketManager;
  vector<uint32_t> m_obs_shape;
  Ptr<NetDevice> m_receivingNetDev;
  uint32_t m_signPacketSize;
  uint32_t m_counterSentPackets;
  uint32_t m_pingPacketIndex = 0;
  int *m_nodes_starting_address;
  
};

}


#endif // PACKET_ROUTING_GYM_ENTITY_H
