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

#ifndef COMPUTE_STATS_V2_H
#define COMPUTE_STATS_V2_H

#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include <vector>

using namespace std;

namespace ns3 {

class ComputeStats
{
public:
  ComputeStats ();
  
  //Average and sum functions
  float getAverage(vector<float> v);
  float getSum(vector<float> v);

  //Converting number of Packets to Bytes
  int transformNbPacketsToBytes(int nbPackets, int size);

  //Adding End-to-end Delay and cost to vectors
  void addE2eDelay(float delay);
  void addCost(float cost);

  //Increment packets Info
  void incrementOverlayPacketsInjected();
  void incrementOverlayPacketsLost();
  void incrementOverlayPacketsArrived();

  void incrementUnderlayPacketsInjected();
  void incrementUnderlayPacketsLost();
  void incrementUnderlayPacketsArrived();

  //Increment global Bytes Info
  void addGlobalBytesData(int value);
  void addGlobalBytesSignaling(int value);

  //Set loss Penalty
  void setLossPenalty(double value);
  void addLossPenaltyToCost();

  
  //Get Functions
  vector<float> getGlobalE2eDelay();
  vector<float> getGlobalCost();
  float getGlobalLossRatio();
  int getGlobalOverlayPacketsInjected();
  int getGlobalOverlayPacketsArrived();
  int getGlobalOverlayPacketsLost();
  int getGlobalOverlayPacketsBuffered();
  int getGlobalUnderlayPacketsInjected();
  int getGlobalUnderlayPacketsArrived();
  int getGlobalUnderlayPacketsLost();
  int getGlobalUnderlayPacketsBuffered();
  vector<float> getLocalE2eDelay();
  vector<float> getLocalCost();
  float getLocalLossRatio();
  int getLocalOverlayPacketsInjected();
  int getLocalOverlayPacketsArrived();
  int getLocalOverlayPacketsLost();
  int getLocalOverlayPacketsBuffered();
  float getSignalingOverhead();


private:
  //Global (static) information
  static vector<float> m_globalE2eDelay;
  static vector<float> m_globalCost;
  static float m_globalLossRatio;

  static int m_globalOverlayPacketsInjected;
  static int m_globalOverlayPacketsArrived;
  static int m_globalOverlayPacketsLost;

  //Information about underlay Packets
  static int m_globalUnderlayPacketsInjected;
  static int m_globalUnderlayPacketsArrived;
  static int m_globalUnderlayPacketsLost;

  static int m_globalBytesData;
  static int m_globalBytesSignaling;

  static double m_lossPenalty;


  //Local (node) Information
  vector<float> m_localE2eDelay;
  vector<float> m_localCost;
  float m_localLossRatio;

  int m_localOverlayPacketsInjected;
  int m_localOverlayPacketsArrived;
  int m_localOverlayPacketsLost;
  
};

}


#endif // PACKET_ROUTING_GYM_ENTITY_H
