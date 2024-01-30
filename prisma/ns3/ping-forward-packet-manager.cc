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
#include "ping-forward-packet-manager.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "point-to-point-net-device.h"
#include "my-tag.h"
#include "ns3/ppp-header.h"
#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"

//#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>



namespace ns3 {

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

NS_LOG_COMPONENT_DEFINE ("PingForwardPacketManager");

PingForwardPacketManager::PingForwardPacketManager ()
{
  PacketManager();
  NS_LOG_FUNCTION (this);
}

PingForwardPacketManager::PingForwardPacketManager (Ptr<Node> node, vector<int> neighbors ) : PacketManager(node, neighbors)
{
  NS_LOG_FUNCTION (this);
}


bool 
PingForwardPacketManager::receivePacket(Ptr<Packet> packet, Ptr<NetDevice> receivingNetDev){
  //Get extra info from packet
    MyTag tagCopy;
  packet->PeekPacketTag(tagCopy);
<<<<<<< HEAD
  // skip transition packets
  if (tagCopy.GetFinalDestination() != m_node->GetId()){
    return false;
  }
=======

>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
  m_lastHop = tagCopy.GetLastHop();
  m_receivingNetDev = receivingNetDev;
  float delay = Simulator::Now().GetSeconds()-(tagCopy.GetStartTime()*0.001);
  uint32_t overlayTunnelIndex = tagCopy.GetTunnelOverlaySendingIndex();
  uint32_t pingPacketIndex = tagCopy.GetOverlayIndex();

  //Send PingBack Packet
  sendPingBackPacket(delay, overlayTunnelIndex, pingPacketIndex);

  
  return true;
}
void
PingForwardPacketManager::sendPingBackPacket(float delay,  uint32_t overlayTunnelIndex, uint32_t pingPacketIndex){
  //Define Tag
  MyTag tagPingBack;

  //Define packet size
  double packetSize=8;
  
  //Setting Packet Tag
  Ptr<Packet> pingBackPckt = Create<Packet> (packetSize);
  tagPingBack.SetSimpleValue(uint8_t(PING_BACK_PACKET));
  tagPingBack.SetFinalDestination(m_lastHop);
  tagPingBack.SetNextHop(m_lastHop);
  tagPingBack.SetLastHop(m_node->GetId());
  tagPingBack.SetSource(m_node->GetId());
  tagPingBack.SetOneHopDelay(delay);
  tagPingBack.SetOverlayIndex(pingPacketIndex);
  tagPingBack.SetTunnelOverlaySendingIndex(overlayTunnelIndex);
 
  //tagPingBack.SetIdValue(m_packetUid);
  tagPingBack.SetTrafficValable(0);

  pingBackPckt->AddPacketTag(tagPingBack);

  //Adding headers
  UdpHeader udp_head;
  pingBackPckt->AddHeader(udp_head);
  
  Ipv4Header ip_head;
  string string_ip_src= "10.2.2."+std::to_string(m_node->GetId()+1);
  Ipv4Address ip_src(string_ip_src.c_str());
  ip_head.SetSource(ip_src);
  string string_ip_dest;
  string_ip_dest= "10.2.2."+std::to_string(m_lastHop+1);
  Ipv4Address ip_dest(string_ip_dest.c_str());
  ip_head.SetDestination(ip_dest);
  ip_head.SetPayloadSize(8+udp_head.GetSerializedSize());
  ip_head.SetProtocol(17);
  pingBackPckt->AddHeader(ip_head);

  
  

  //Send the sign packet
  m_receivingNetDev->Send(pingBackPckt, m_destAddr, 0x800);
}


string
PingForwardPacketManager::getInfo()
{
  string myInfo = PacketManager::getInfo();
  myInfo += ", NN Index="; //18
  myInfo += std::to_string(m_NNIndex);
  myInfo += ", segment Index="; //19
  myInfo += std::to_string(m_segIndex);
  myInfo += ", NodeId Signaled="; //20
  myInfo += std::to_string(m_source); 
  
  return myInfo;
}

}// ns3 namespace