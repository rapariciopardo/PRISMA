/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo and Lucile Sassatelli
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
 * This work is partially based on Copyright (c) 2008 INRIA
 * available on https://www.nsnam.org/doxygen/on-off-helper_8cc_source.html
 */
#include "ospf-signaling-app-helper.h"
#include "ospf-signaling-application.h"
#include "ns3/inet-socket-address.h"
#include "ns3/packet-socket-address.h"
#include "ns3/string.h"
#include "ns3/data-rate.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/names.h"
#include "ns3/random-variable-stream.h"
//#include "ns3/onoff-application.h"

namespace ns3 {

OspfSignalingAppHelper::OspfSignalingAppHelper (std::string protocol, Address address)
{
  m_factory.SetTypeId ("ns3::OspfSignalingGeneratorApplication");
  m_factory.Set ("Protocol", StringValue (protocol));
  m_factory.Set ("Remote", AddressValue (address));
}

void 
OspfSignalingAppHelper::SetAttribute (std::string name, const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
OspfSignalingAppHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
OspfSignalingAppHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
OspfSignalingAppHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}

Ptr<Application>
OspfSignalingAppHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Application> app = m_factory.Create<Application> ();
  node->AddApplication (app);

  return app;
}


void 
OspfSignalingAppHelper::SetAverageStep (float syncStep, uint32_t packetSize)
{
  m_factory.Set ("SyncStep", DoubleValue(syncStep));
  m_factory.Set ("AvgPacketSize", UintegerValue (packetSize));
}

void 
OspfSignalingAppHelper::SetNNeighbors (uint32_t n_neighbors)
{
  m_factory.Set ("n_neighbors", UintegerValue (n_neighbors));
}

void
OspfSignalingAppHelper::SetSourceDestInterface (uint32_t src, uint32_t dest, uint32_t interface){
  m_factory.Set ("src", UintegerValue(src));
  m_factory.Set ("dest", UintegerValue(dest));
  m_factory.Set ("interface", UintegerValue(interface));
}
} // namespace ns3