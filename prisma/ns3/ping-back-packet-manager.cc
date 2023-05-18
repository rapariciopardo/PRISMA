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
#include "ping-back-packet-manager.h"
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

NS_LOG_COMPONENT_DEFINE ("PingBackPacketManager");

PingBackPacketManager::PingBackPacketManager ()
{
  PacketManager();
  NS_LOG_FUNCTION (this);
}

PingBackPacketManager::PingBackPacketManager (Ptr<Node> node, vector<int> neighbors ) : PacketManager(node, neighbors)
{
  NS_LOG_FUNCTION (this);
}

void
PingBackPacketManager::setMovingAverageSize(uint32_t value){
  m_movingAverageSize = value;
}

void 
PingBackPacketManager::addSentPingForwardPacket(uint64_t id, uint64_t start_time){
  NS_LOG_UNCOND("PingBackPacketManager::addSentPingForwardPacket");
  SentPacket sentPacket;
  sentPacket.start_time = start_time;
  sentPacket.uid = id;
  sentPacket.type = PING_FORWARD_PACKET;
  for(size_t i = 0;i<m_neighbors.size();i++){
    m_sentPingForwardPackets[i].push_back(sentPacket);
  }
  
}

float
PingBackPacketManager::getMaxTimePingForwardPacketSent(uint32_t index){
  NS_LOG_UNCOND("PingBackPacketManager::getMaxTimePingForwardPacketSent");
  if(m_sentPingForwardPackets[index].size()>0){
    return Simulator::Now().GetSeconds() - m_sentPingForwardPackets[index][0].start_time*0.001;
  } else{
    return 0.0;
  }
} 


bool 
PingBackPacketManager::receivePacket(Ptr<Packet> packet, Ptr<NetDevice> receivingNetDev){
  NS_LOG_UNCOND("PingBackPacketManager::receivePacket");
  //Get extra info from packet
  MyTag tagCopy;
  packet->PeekPacketTag(tagCopy); 

  float delay = tagCopy.GetOneHopDelay();
  m_overlayTunnelIndex = tagCopy.GetTunnelOverlaySendingIndex();
  m_pingPacketIndex = tagCopy.GetOverlayIndex();
  
  //Erasing sent packets which were acked
  auto it = m_sentPingForwardPackets[m_overlayTunnelIndex].begin();
  NS_LOG_UNCOND("PingBackPacketManager::receivePacket 2.1 empt" << int(m_sentPingForwardPackets[m_overlayTunnelIndex].empty()) << " " << m_tunnelsDelay[m_overlayTunnelIndex].size());
  if (m_sentPingForwardPackets[m_overlayTunnelIndex].empty()==0){
    while(it->uid != m_pingPacketIndex){
      it = m_sentPingForwardPackets[m_overlayTunnelIndex].erase(it);
    }
    it = m_sentPingForwardPackets[m_overlayTunnelIndex].erase(it);
  }
  //Adding tunnel delay in a circular array
  if(m_tunnelsDelay[m_overlayTunnelIndex].size()>=m_movingAverageSize){
    assert(!m_tunnelsDelay[m_overlayTunnelIndex].empty());
    m_tunnelsDelay[m_overlayTunnelIndex].erase(m_tunnelsDelay[m_overlayTunnelIndex].begin());
  }
    NS_LOG_UNCOND("PingBackPacketManager::receivePacket 4");
  m_tunnelsDelay[m_overlayTunnelIndex].push_back(delay);
    NS_LOG_UNCOND("PingBackPacketManager::receivePacket 5");
  
  return true;
}

string
PingBackPacketManager::getInfo()
{
  string myInfo = PacketManager::getInfo();
  
  return myInfo;
}

}// ns3 namespace