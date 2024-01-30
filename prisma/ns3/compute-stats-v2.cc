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
#include "compute-stats-v2.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "point-to-point-net-device.h"
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


using namespace std;
namespace ns3 {





NS_LOG_COMPONENT_DEFINE ("ComputeStats");

vector<float> ComputeStats::m_globalE2eDelay;
vector<float> ComputeStats::m_globalCost;
vector<float> ComputeStats::m_globalUnderlayE2eDelay;
vector<float> ComputeStats::m_globalUnderlayCost;
float ComputeStats::m_globalLossRatio;
int ComputeStats::m_globalOverlayPacketsInjected;
int ComputeStats::m_globalOverlayPacketsArrived;
int ComputeStats::m_globalOverlayPacketsLost;
<<<<<<< HEAD
int ComputeStats::m_globalOverlayPacketsRejected;
int ComputeStats::m_globalUnderlayPacketsInjected;
int ComputeStats::m_globalUnderlayPacketsArrived;
int ComputeStats::m_globalUnderlayPacketsLost;
int ComputeStats::m_globalUnderlayPacketsRejected;
=======
int ComputeStats::m_globalUnderlayPacketsInjected;
int ComputeStats::m_globalUnderlayPacketsArrived;
int ComputeStats::m_globalUnderlayPacketsLost;
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
int ComputeStats::m_globalBytesData;
int ComputeStats::m_globalBytesSignaling;
double ComputeStats::m_lossPenalty;


//NS_OBJECT_ENSURE_REGISTERED (ComputeStats);

ComputeStats::ComputeStats ()
{
  NS_LOG_FUNCTION (this);
  
  //Ptr<PointToPointNetDevice> m_array_p2pNetDevs[m_num_nodes][m_num_nodes];
  
  //std::vector<std::vector<int>> matrix(RR, vector<int>(CC);

  
}
float ComputeStats::getAverage(vector<float> v) {
  if (v.empty()) {
    return 0.0;
  }

  float sum = 0.0;
  for (float i: v) {
    sum += (float)i;
  }
  return sum / v.size();
}

float ComputeStats::getSum(vector<float> v) {
  if (v.empty()) {
    return 0.0;
  }

  float sum = 0.0;
  for (float i: v) {
    sum += (float)i;
  }
  return sum;
}

int ComputeStats::transformNbPacketsToBytes(int nbPackets, int size){
    return nbPackets*size;
}

void ComputeStats::addE2eDelay(float delay){
    m_localE2eDelay.push_back(delay);
    m_globalE2eDelay.push_back(delay);
}

void ComputeStats::addCost(float cost){
    m_localCost.push_back(cost);
    m_globalCost.push_back(cost);
}

void ComputeStats::addUnderlayE2eDelay(float delay){
    m_globalUnderlayE2eDelay.push_back(delay);
}

void ComputeStats::addUnderlayCost(float cost){
    m_globalUnderlayCost.push_back(cost);
}


void ComputeStats::addLossPenaltyToCost(){
    m_localCost.push_back(m_lossPenalty);
    m_globalCost.push_back(m_lossPenalty);
}

void ComputeStats::addLossPenaltyToUnderlayCost(){
    m_globalUnderlayCost.push_back(m_lossPenalty);
}


void ComputeStats::incrementOverlayPacketsInjected(){
    m_localOverlayPacketsInjected += 1;
    m_globalOverlayPacketsInjected += 1;
}

void ComputeStats::incrementOverlayPacketsArrived(){
    m_localOverlayPacketsArrived += 1;
    m_globalOverlayPacketsArrived += 1;
}

void ComputeStats::incrementOverlayPacketsLost(){
    m_localOverlayPacketsLost += 1;
    m_globalOverlayPacketsLost += 1;
}

<<<<<<< HEAD
void ComputeStats::incrementOverlayPacketsRejected(){
    m_globalOverlayPacketsRejected += 1;
}

=======
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
void ComputeStats::incrementUnderlayPacketsInjected(){
    m_globalUnderlayPacketsInjected += 1;
}

void ComputeStats::incrementUnderlayPacketsArrived(){
    m_globalUnderlayPacketsArrived += 1;
}

void ComputeStats::incrementUnderlayPacketsLost(){
    m_globalUnderlayPacketsLost += 1;
}

<<<<<<< HEAD
void ComputeStats::incrementUnderlayPacketsRejected(){
    m_globalUnderlayPacketsRejected += 1;
}

=======
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
void ComputeStats::addGlobalBytesData(int value){
    m_globalBytesData += value;
}

void ComputeStats::addGlobalBytesSignaling(int value){
    m_globalBytesSignaling += value;
}

void ComputeStats::setLossPenalty(double value){
    m_lossPenalty = value;
}

vector<float> ComputeStats::getGlobalE2eDelay(){
    return m_globalE2eDelay;
}

vector<float> ComputeStats::getGlobalCost(){
    return m_globalCost;
}

vector<float> ComputeStats::getGlobalUnderlayE2eDelay(){
    return m_globalUnderlayE2eDelay;
}

vector<float> ComputeStats::getGlobalUnderlayCost(){
    return m_globalUnderlayCost;
}

float ComputeStats::getGlobalLossRatio(){
    return m_globalLossRatio;
}

int ComputeStats::getGlobalOverlayPacketsInjected(){
    return m_globalOverlayPacketsInjected;
}

int ComputeStats::getGlobalOverlayPacketsArrived(){
    return m_globalOverlayPacketsArrived;
}

int ComputeStats::getGlobalOverlayPacketsLost(){
    return m_globalOverlayPacketsLost;
}

<<<<<<< HEAD
int ComputeStats::getGlobalOverlayPacketsRejected(){
    return m_globalOverlayPacketsRejected;
}

=======
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
int ComputeStats::getGlobalOverlayPacketsBuffered(){
    return m_globalOverlayPacketsInjected - (m_globalOverlayPacketsArrived + m_globalOverlayPacketsLost);
}

int ComputeStats::getGlobalUnderlayPacketsInjected(){
    return m_globalUnderlayPacketsInjected;
}

int ComputeStats::getGlobalUnderlayPacketsArrived(){
    return m_globalUnderlayPacketsArrived;
}

int ComputeStats::getGlobalUnderlayPacketsLost(){
    return m_globalUnderlayPacketsLost;
}

<<<<<<< HEAD
int ComputeStats::getGlobalUnderlayPacketsRejected(){
    return m_globalUnderlayPacketsRejected;
}

=======
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
int ComputeStats::getGlobalUnderlayPacketsBuffered(){
    return m_globalUnderlayPacketsInjected - (m_globalUnderlayPacketsArrived + m_globalUnderlayPacketsLost);
}

vector<float> ComputeStats::getLocalE2eDelay(){
    return m_localE2eDelay;
}

vector<float> ComputeStats::getLocalCost(){
    return m_localCost;
}

float ComputeStats::getLocalLossRatio(){
    return m_localLossRatio;
}

int ComputeStats::getLocalOverlayPacketsInjected(){
    return m_localOverlayPacketsInjected;
}

int ComputeStats::getLocalOverlayPacketsArrived(){
    return m_localOverlayPacketsArrived;
}

int ComputeStats::getLocalOverlayPacketsLost(){
    return m_localOverlayPacketsLost;
}

int ComputeStats::getLocalOverlayPacketsBuffered(){
    return m_localOverlayPacketsInjected - m_localOverlayPacketsArrived - m_localOverlayPacketsLost;
}

float ComputeStats::getSignalingOverhead(){
    if(m_globalBytesData==0) return 0.0;
    return float(m_globalBytesSignaling)/float(m_globalBytesData);
}
}// ns3 namespace