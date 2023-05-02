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
#include "packet-manager.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/ppp-header.h"
#include "point-to-point-net-device.h"
#include "my-tag.h"
#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"

#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

namespace ns3 {



NS_LOG_COMPONENT_DEFINE ("PacketManager");

PacketManager::PacketManager ()
{
  NS_LOG_FUNCTION (this);
}


PacketManager::PacketManager (Ptr<Node> node, vector<int> neighbors)
{
  NS_LOG_FUNCTION (this);
  m_node = node;
  m_neighbors = neighbors;
  m_computeStats = new ComputeStats();
}
void PacketManager::step(Ptr<Packet> packet){
    this->receivePacket(packet);
    this->getInfo();
}

void PacketManager::receivePacket(Ptr<Packet> packet){
    //Packet copy
    Ptr<Packet> p;
    p = packet->Copy();

    //Packet's tag
    MyTag tagCopy;
    p->PeekPacketTag(tagCopy);
    
    //Packet's Header
    p->RemoveHeader(m_packetPppHeader);  
    p->RemoveHeader(m_packetIpHeader);
    p->RemoveHeader(m_packetUdpHeader);

    //Get packet copy
    m_packet = p->Copy();


    //Get Packet Information
    m_lastHop = tagCopy.GetLastHop();
    m_nextHop = tagCopy.GetNextHop();
    m_source = tagCopy.GetSource();
    m_destination = tagCopy.GetFinalDestination();
    m_sourceTimeStamp = tagCopy.GetStartTime();
    m_packetSize = packet->GetSize();
    m_packetUid = packet->GetUid();
    m_packetType = PacketType(tagCopy.GetSimpleValue());

    if(m_source==m_node->GetId()){
      m_arrivedAtOrigin = true;
    } else{
      m_arrivedAtOrigin = false;
    }

    if(m_destination==m_node->GetId()){
      m_arrivedAtFinalDest = true;
    } else{
      m_arrivedAtFinalDest = false;
    }


}

string PacketManager::getInfo(){

    std::string myInfo = "End to End Delay="; //0
    myInfo += std::to_string(Simulator::Now().GetSeconds()-m_sourceTimeStamp);

    myInfo += ", Packet Size="; //1
    myInfo += std::to_string(m_packetSize);
    
    myInfo += ", Current sim time ="; //2
    myInfo += std::to_string(Simulator::Now().GetSeconds());
    
    myInfo += ", Pkt ID ="; //3
    myInfo += std::to_string(m_packetUid);

     myInfo += ", packetType ="; //4
    myInfo += std::to_string(m_packetType);

    myInfo += ", Avg End to End Delay ="; //5
    myInfo += std::to_string(m_computeStats->getAverage(m_computeStats->getGlobalE2eDelay()));

    myInfo += ", Avg Cost ="; //6
    myInfo += std::to_string(m_computeStats->getAverage(m_computeStats->getGlobalCost()));

    myInfo += ", Packets dropped ="; //7
    myInfo += std::to_string(m_computeStats->getGlobalOverlayPacketsLost());

    myInfo += ", Packets delivered ="; //8
    myInfo += std::to_string(m_computeStats->getGlobalOverlayPacketsArrived());

    myInfo += ", Packets injected ="; //9
    myInfo += std::to_string(m_computeStats->getGlobalOverlayPacketsInjected());

    myInfo += ",Packets Buffered ="; //10
    myInfo += std::to_string(m_computeStats->getGlobalOverlayPacketsBuffered());

    myInfo += ", Packets dropped Underlay ="; //11
    myInfo += std::to_string(m_computeStats->getGlobalUnderlayPacketsLost());

    myInfo += ", Packets delivered Underlay="; //12
    myInfo += std::to_string(m_computeStats->getGlobalUnderlayPacketsArrived());

    myInfo += ", Packets injected Underlay="; //13
    myInfo += std::to_string(m_computeStats->getGlobalUnderlayPacketsInjected());

    myInfo += ",Packets Buffered Underlay="; //14
    myInfo += std::to_string(m_computeStats->getGlobalUnderlayPacketsBuffered());
    
    myInfo += ",Signaling overhead ="; //15
    myInfo += std::to_string(m_computeStats->getSignalingOverhead());

    return myInfo;
   
}

void 
PacketManager::setNetDevContainer(NetDeviceContainer netDevs){
  m_netDevs = netDevs;
}

}// ns3 namespace