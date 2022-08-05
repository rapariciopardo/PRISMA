/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- *//* * Copyright (c) 2006,2007 INRIA * * This program is free software; you can redistribute it and/or modify * it under the terms of the GNU General Public License version 2 as * published by the Free Software Foundation; * * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */


#include "my-tag.h"
#include <iostream>

using namespace ns3;


TypeId 
MyTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MyTag")
    .SetParent<Tag> ()
    .AddConstructor<MyTag> ()
    .AddAttribute ("SimpleValue",
                   "A simple value",
                   EmptyAttributeValue (),
                   MakeUintegerAccessor (&MyTag::GetSimpleValue),
                   MakeUintegerChecker<uint8_t> ())
  ;
  return tid;
}
TypeId 
MyTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t 
MyTag::GetSerializedSize (void) const
{
  uint32_t ret = 9;
  if(m_simpleValue==0){
    ret += 9;
  }
  else if(m_simpleValue==1){
    ret += 12;
  }
  else if(m_simpleValue==2){
    ret += 8;
  }
  else if (m_simpleValue==3){
    ret += 12;
  }
  else if (m_simpleValue==4){
    ret += 12;
  }
  return ret;
}
void 
MyTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_simpleValue);
  i.WriteU32(m_finalDestination);
  i.WriteU32 (m_lastHop);
  if(m_simpleValue==0){
    i.WriteU64 (m_startTime);
    i.WriteU8 (m_trafficValable);
  }
  else if(m_simpleValue==1){
    i.WriteU32 (m_segIndex);
    i.WriteU32 (m_NNIndex);
    i.WriteU32 (m_nodeId);
  }
  else if(m_simpleValue==2){
    i.WriteU64 (m_pktId);
  }
  else if(m_simpleValue==3){
    i.WriteU64 (m_startTime);
    i.WriteU32 (m_overlayIndex);
  }
  else if(m_simpleValue==4){
    i.WriteU64 (m_startTime);
    i.WriteU32 (m_overlayIndex);
  }
  
  
}
void 
MyTag::Deserialize (TagBuffer i)
{
  m_simpleValue = i.ReadU8 ();
  m_finalDestination = i.ReadU32 ();
  m_lastHop = i.ReadU32 ();
  if(m_simpleValue==0){
    m_startTime=i.ReadU64 ();
    m_trafficValable =i.ReadU8 ();
  }
  else if(m_simpleValue==1){
    m_segIndex=i.ReadU32 ();
    m_NNIndex=i.ReadU32 ();
    m_nodeId=i.ReadU32 ();
  }
  else if(m_simpleValue==2){
    m_pktId=i.ReadU64 ();
  }
  else if(m_simpleValue==3){
    m_startTime=i.ReadU64 ();
    m_overlayIndex=i.ReadU32 ();
  }
  else if(m_simpleValue==4){
    m_startTime=i.ReadU64 ();
    m_overlayIndex = i.ReadU32 ();
  }
 
}
void 
MyTag::Print (std::ostream &os) const
{
  os << "v=" << (uint32_t)m_simpleValue << "   "<<(uint32_t)m_pktId<<"    "<<m_segIndex<<"     "<<m_NNIndex;
}
void 
MyTag::SetSimpleValue (uint8_t value)
{
  m_simpleValue = value;
}
uint8_t 
MyTag::GetSimpleValue (void) const
{
  return m_simpleValue;
}
void
MyTag::SetIdValue (uint64_t value)
{
  m_pktId = value;
}
uint64_t 
MyTag::GetIdValue (void) const
{
  return m_pktId;
}
void
MyTag::SetSegIndex (uint32_t value)
{
  m_segIndex = value;
}
uint32_t 
MyTag::GetSegIndex (void) const
{
  return m_segIndex;
}
void
MyTag::SetNNIndex (uint32_t value)
{
  m_NNIndex = value;
}
uint32_t 
MyTag::GetNNIndex (void) const
{
  return m_NNIndex;
}
void
MyTag::SetNodeId (uint32_t value)
{
  m_nodeId = value;
}
uint32_t 
MyTag::GetNodeId (void) const
{
  return m_nodeId;
}
void
MyTag::SetStartTime (uint64_t value)
{
  m_startTime = value;
}
uint64_t 
MyTag::GetStartTime (void) const
{
  return m_startTime;
}
void
MyTag::SetFinalDestination (uint32_t value)
{
  m_finalDestination = value;
}
uint32_t 
MyTag::GetFinalDestination (void) const
{
  return m_finalDestination;
}
void
MyTag::SetLastHop (uint32_t value)
{
  m_lastHop = value;
}
uint32_t 
MyTag::GetLastHop (void) const
{
  return m_lastHop;
}
void 
MyTag::SetTrafficValable (uint8_t value)
{
  m_trafficValable = value;
}
uint8_t 
MyTag::GetTrafficValable (void) const
{
  return m_trafficValable;
}
void 
MyTag::SetOverlayIndex (uint32_t value)
{
  m_overlayIndex = value;
}
uint32_t 
MyTag::GetOverlayIndex (void) const
{
  return m_overlayIndex;
}

