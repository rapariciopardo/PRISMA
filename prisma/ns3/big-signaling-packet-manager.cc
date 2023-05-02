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
#include "big-signaling-packet-manager.h"
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

NS_LOG_COMPONENT_DEFINE ("BigSignalingPacketManager");

BigSignalingPacketManager::BigSignalingPacketManager ()
{
  PacketManager();
  NS_LOG_FUNCTION (this);
}

BigSignalingPacketManager::BigSignalingPacketManager (Ptr<Node> node, vector<int> neighbors ) : PacketManager(node, neighbors)
{
  NS_LOG_FUNCTION (this);
}


bool 
BigSignalingPacketManager::receivePacket(Ptr<Packet> packet){
  PacketManager::receivePacket(packet);

  //Check if the node is not the packet source
  if(m_source == m_node->GetId()){
    return false;
  }

  //Get extra info from packet
  MyTag tagCopy;
  m_packet->PeekPacketTag(tagCopy);
  m_NNIndex = tagCopy.GetNNIndex();
  m_segIndex = tagCopy.GetSegIndex();
  
  return true;
}

string
BigSignalingPacketManager::getInfo()
{
  string myInfo = PacketManager::getInfo();
  myInfo += ", NN Index="; //16
  myInfo += std::to_string(m_NNIndex);
  myInfo += ", segment Index="; //17
  myInfo += std::to_string(m_segIndex);
  myInfo += ", NodeId Signaled="; //18
  myInfo += std::to_string(m_source); 
  
  return myInfo;
}

}// ns3 namespace