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
 * This work is parially based on Copyright (c) 2006 Georgia Tech Research Corporation
 * available on https://www.nsnam.org/doxygen/onoff-application_8cc_source.html
 */


#include "ns3/log.h"
#include "ns3/address.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/packet-socket-address.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/data-rate.h"
#include "ns3/random-variable-stream.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/trace-source-accessor.h"
#include "big-signaling-application.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "my-tag.h"
namespace ns3 {


NS_LOG_COMPONENT_DEFINE ("BigSignalingGeneratorApplication");

NS_OBJECT_ENSURE_REGISTERED (BigSignalingGeneratorApplication);

TypeId
BigSignalingGeneratorApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BigSignalingGeneratorApplication")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<BigSignalingGeneratorApplication> ()
    .AddAttribute ("AvgPacketSize", "The average size of packets sent",
                   UintegerValue (36000),
                   MakeUintegerAccessor (&BigSignalingGeneratorApplication::m_pktSizeMean),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("src", "src",
                   UintegerValue (0),
                   MakeUintegerAccessor (&BigSignalingGeneratorApplication::m_src),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("srcOverlay", "srcOverlay",
                   UintegerValue (0),
                   MakeUintegerAccessor (&BigSignalingGeneratorApplication::m_srcOverlay),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("dest", "dest",
                   UintegerValue (1),
                   MakeUintegerAccessor (&BigSignalingGeneratorApplication::m_dest),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("SyncStep", "The frequency the big signaling is sent, Synchronized in seconds.",
                   DoubleValue(0.5),
                   MakeDoubleAccessor (&BigSignalingGeneratorApplication::m_syncStep),
                   MakeDoubleChecker<double>())
    .AddAttribute ("Remote", "The address of the destination",
                   AddressValue (),
                   MakeAddressAccessor (&BigSignalingGeneratorApplication::m_peer),
                   MakeAddressChecker ())
    .AddAttribute ("MaxBytes", 
                   "The total number of bytes to send. Once these bytes are sent, "
                   "no packet is sent again. The value zero means "
                   "that there is no limit.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&BigSignalingGeneratorApplication::m_maxBytes),
                   MakeUintegerChecker<uint64_t> ())
    .AddAttribute ("Protocol", "The type of protocol to use. This should be "
                   "a subclass of ns3::SocketFactory",
                   TypeIdValue (TcpSocketFactory::GetTypeId ()),
                   MakeTypeIdAccessor (&BigSignalingGeneratorApplication::m_tid),
                   // This should check for SocketFactory as a parent
                   MakeTypeIdChecker ())
    .AddTraceSource ("Tx", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&BigSignalingGeneratorApplication::m_txTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("TxWithAddresses", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&BigSignalingGeneratorApplication::m_txTraceWithAddresses),
                     "ns3::Packet::TwoAddressTracedCallback")
  ;
  return tid;
}


BigSignalingGeneratorApplication::BigSignalingGeneratorApplication ()
  : m_socket (0),
    m_connected (false),
    m_lastStartTime (Seconds (0)),
    m_totBytes (0)
{
  NS_LOG_FUNCTION (this);
}

BigSignalingGeneratorApplication::~BigSignalingGeneratorApplication()
{
  NS_LOG_FUNCTION (this);
}

void 
BigSignalingGeneratorApplication::SetMaxBytes (uint64_t maxBytes)
{
  NS_LOG_FUNCTION (this << maxBytes);
  m_maxBytes = maxBytes;
}

Ptr<Socket>
BigSignalingGeneratorApplication::GetSocket (void) const
{
  NS_LOG_FUNCTION (this);
  return m_socket;
}

void
BigSignalingGeneratorApplication::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  m_socket = 0;
  // chain up
  Application::DoDispose ();
}

// Application Methods
void BigSignalingGeneratorApplication::StartApplication () // Called at time specified by Start
{
  NS_LOG_FUNCTION (this);

  // Create the socket if not already
  if (!m_socket)
    {
      NS_LOG_UNCOND(m_tid);
      m_socket = Socket::CreateSocket (GetNode (), m_tid);
      if (Inet6SocketAddress::IsMatchingType (m_peer))
        {
          if (m_socket->Bind6 () == -1)
            {
              NS_FATAL_ERROR ("Failed to bind socket");
            }
        }
      else if (InetSocketAddress::IsMatchingType (m_peer) ||
               PacketSocketAddress::IsMatchingType (m_peer))
        {
          if (m_socket->Bind () == -1)
            {
              NS_FATAL_ERROR ("Failed to bind socket");
            }
        }
      m_socket->Connect (m_peer);
      m_socket->SetAllowBroadcast (true);
      m_socket->ShutdownRecv ();

      m_socket->SetConnectCallback (
        MakeCallback (&BigSignalingGeneratorApplication::ConnectionSucceeded, this),
        MakeCallback (&BigSignalingGeneratorApplication::ConnectionFailed, this));
    }
  m_avgRateFailSafe = m_avgRate;

  // Insure no pending event
  CancelEvents ();
  // If we are not yet connected, there is nothing to do here
  // The ConnectionComplete upcall will start timers at that time
  //if (!m_connected) return;
  //ScheduleStartEvent ();
  StartSending();
}

void BigSignalingGeneratorApplication::StopApplication () // Called at time specified by Stop
{
  NS_LOG_FUNCTION (this);

  CancelEvents ();
  if(m_socket != 0)
    {
      m_socket->Close ();
    }
  else
    {
      NS_LOG_WARN ("BigSignalingGeneratorApplication found null socket to close in StopApplication");
    }
}

void BigSignalingGeneratorApplication::CancelEvents ()
{
  NS_LOG_FUNCTION (this);
  m_avgRateFailSafe = m_avgRate;
  Simulator::Cancel (m_sendEvent);
}


// Event handlers
void BigSignalingGeneratorApplication::StartSending ()
{
  NS_LOG_FUNCTION (this);
  m_lastStartTime = Simulator::Now ();
  //NS_LOG_UNCOND("NET DEVICE "<<m_socket->GetBoundNetDevice()->GetNode()->GetId());
  m_NNIndex = 0;
  m_segIndex = 0;
  ScheduleNextTx ();  // Schedule the send packet event
}

// Private helpers
void BigSignalingGeneratorApplication::ScheduleNextTx ()
{
  NS_LOG_FUNCTION (this);

  if (m_maxBytes == 0 || m_totBytes < m_maxBytes)
    {
      m_segIndex++;
      m_segSize = 512;
      if(m_segIndex>=uint32_t(m_pktSizeMean/m_segSize)){
        m_NNIndex++;
        m_segIndex=0;
      }
      double dataRate = (m_pktSizeMean*8)/m_syncStep;
      double delay = ((m_segSize*8)/dataRate);
      //iat->GetValue(); // bits/ static_cast<double>(m_avgRate.GetBitRate ());
      Time nextTime (Seconds (delay)); // Time till next packet
      //NS_LOG_LOGIC ("nextTime = " << nextTime);
      m_sendEvent = Simulator::Schedule (nextTime,
                                         &BigSignalingGeneratorApplication::SendPacket, this);
    }
  else
    { // All done, cancel any pending events
      StopApplication ();
    }
}

void BigSignalingGeneratorApplication::SendPacket ()
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT (m_sendEvent.IsExpired ());
  m_pktSize = m_pktSizeMean;
  Ptr<Packet> packet = Create<Packet> (m_segSize);
  MyTag tag;
  tag.SetSimpleValue(0x01);
  tag.SetTrafficValable(0);
  tag.SetFinalDestination(m_dest-1);
  tag.SetSource(m_src-1);
  tag.SetNextHop(m_dest-1);
  tag.SetLastHop(1000);//(m_src-1);
  tag.SetSegIndex(m_segIndex);
  tag.SetNNIndex(m_NNIndex);
  // if(m_srcOverlay-1>11) NS_LOG_UNCOND("ERROR "<<m_src-1);
  tag.SetNodeId(m_srcOverlay-1);
  packet->AddPacketTag(tag);
  m_txTrace (packet);
  std::string start_time = std::to_string(Simulator::Now().GetMilliSeconds());
  m_socket->SetIpTtl(255);
  m_socket->Send(packet);
  m_totBytes += m_segSize;
  Address localAddress;
  m_socket->GetSockName (localAddress);
  if (InetSocketAddress::IsMatchingType (m_peer))
    {
      NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds ()
                   << "s on-off application sent "
                   <<  packet->GetSize () << " bytes to "
                   << InetSocketAddress::ConvertFrom(m_peer).GetIpv4 ()
                   << " port " << InetSocketAddress::ConvertFrom (m_peer).GetPort ()
                   << " total Tx " << m_totBytes << " bytes");
      m_txTraceWithAddresses (packet, localAddress, m_peer);
    }
  else if (Inet6SocketAddress::IsMatchingType (m_peer))
    {
      NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds ()
                   << "s on-off application sent "
                   <<  packet->GetSize () << " bytes to "
                   << Inet6SocketAddress::ConvertFrom(m_peer).GetIpv6 ()
                   << " port " << Inet6SocketAddress::ConvertFrom (m_peer).GetPort ()
                   << " total Tx " << m_totBytes << " bytes");
      m_txTraceWithAddresses (packet, localAddress, Inet6SocketAddress::ConvertFrom(m_peer));
    }
  m_lastStartTime = Simulator::Now ();
  ScheduleNextTx ();
}


void BigSignalingGeneratorApplication::ConnectionSucceeded (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  m_connected = true;
}

void BigSignalingGeneratorApplication::ConnectionFailed (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
}


} // Namespace ns3
