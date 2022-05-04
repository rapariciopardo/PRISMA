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
    .AddAttribute ("IdValue",
                   "A simple Id value",
                   EmptyAttributeValue (),
                   MakeUintegerAccessor (&MyTag::GetIdValue),
                   MakeUintegerChecker<uint64_t> ())
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
  return 10;
}
void 
MyTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_simpleValue);
  i.WriteU64 (m_pktId);
}
void 
MyTag::Deserialize (TagBuffer i)
{
  m_simpleValue = i.ReadU8 ();
  m_pktId = i.ReadU64 ();
}
void 
MyTag::Print (std::ostream &os) const
{
  os << "v=" << (uint32_t)m_simpleValue << "   "<<(uint32_t)m_pktId;
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