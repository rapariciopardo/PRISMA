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
<<<<<<< HEAD
PingBackPacketManager::setTunnelsMaxDelays(vector<vector<double>> tunnelsMaxDelays){
  m_tunnelsMaxDelays = tunnelsMaxDelays;
}

void 
=======
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
PingBackPacketManager::addSentPingForwardPacket(uint64_t id, uint64_t start_time){
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
  if(m_sentPingForwardPackets[index].size()>0){
<<<<<<< HEAD
    return std::min((Simulator::Now().GetSeconds() - m_sentPingForwardPackets[index][0].start_time*0.001)/2.0, m_tunnelsMaxDelays[m_map_overlay_array[m_node->GetId()]][index]);
=======
    return std::min(Simulator::Now().GetSeconds() - m_sentPingForwardPackets[index][0].start_time*0.001, 2.60);
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
  } else{
    return 0.0;
  }
} 


bool 
PingBackPacketManager::receivePacket(Ptr<Packet> packet, Ptr<NetDevice> receivingNetDev){
  //Get extra info from packet
  MyTag tagCopy;
  packet->PeekPacketTag(tagCopy); 
<<<<<<< HEAD
  if (tagCopy.GetFinalDestination() != m_node->GetId()) {
    return false;
  }
=======
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00

  float delay = tagCopy.GetOneHopDelay();
  m_overlayTunnelIndex = tagCopy.GetTunnelOverlaySendingIndex();
  m_pingPacketIndex = tagCopy.GetOverlayIndex();
<<<<<<< HEAD
  if (m_pingPacketIndex == 0){
    NS_LOG_UNCOND( m_map_overlay_array[m_node->GetId()] << " " << m_overlayTunnelIndex << " " << delay);
  }
  // remove the packets until the one that was acked
  if (m_sentPingForwardPackets[m_overlayTunnelIndex].begin()->uid == m_pingPacketIndex){
    m_sentPingForwardPackets[m_overlayTunnelIndex].erase(m_sentPingForwardPackets[m_overlayTunnelIndex].begin());
  }
  else{
    while(m_sentPingForwardPackets[m_overlayTunnelIndex].begin()->uid != m_pingPacketIndex){
      m_sentPingForwardPackets[m_overlayTunnelIndex].erase(m_sentPingForwardPackets[m_overlayTunnelIndex].begin());
    }
    m_sentPingForwardPackets[m_overlayTunnelIndex].erase(m_sentPingForwardPackets[m_overlayTunnelIndex].begin());


  }
  


=======
  
  //Erasing sent packets which were acked
  for (auto it = m_sentPingForwardPackets[m_overlayTunnelIndex].begin(); it != m_sentPingForwardPackets[m_overlayTunnelIndex].end(); ++it){
    if (it->uid == m_pingPacketIndex){
      m_sentPingForwardPackets[m_overlayTunnelIndex].erase(it);
      break;
    }
  }
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
  //Adding tunnel delay in a circular array
  if(m_tunnelsDelay[m_overlayTunnelIndex].size()>=m_movingAverageSize){
    assert(!m_tunnelsDelay[m_overlayTunnelIndex].empty());
    m_tunnelsDelay[m_overlayTunnelIndex].erase(m_tunnelsDelay[m_overlayTunnelIndex].begin());
  }
  m_tunnelsDelay[m_overlayTunnelIndex].push_back(delay);
  
  return true;
}

string
PingBackPacketManager::getInfo()
{
  string myInfo = PacketManager::getInfo();
  
  return myInfo;
}

}// ns3 namespace