/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: George F. Riley<riley@ece.gatech.edu>
//

// ns3 - On/Off Data Source Application class
// George F. Riley, Georgia Tech, Spring 2007
// Adapted from ApplicationOnOff in GTNetS.

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
#include "ns3/double.h"
#include "ns3/trace-source-accessor.h"
#include "poisson-application.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/string.h"
#include "ns3/pointer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PoissonGeneratorApplication");

NS_OBJECT_ENSURE_REGISTERED (PoissonGeneratorApplication);

TypeId
PoissonGeneratorApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PoissonGeneratorApplication")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<PoissonGeneratorApplication> ()
    .AddAttribute ("AvgDataRate", "The flow average data rate.",
                   DataRateValue (DataRate ("500kb/s")),
                   MakeDataRateAccessor (&PoissonGeneratorApplication::m_avgRate),
                   MakeDataRateChecker ())
    .AddAttribute ("AvgPacketSize", "The average size of packets sent",
                   UintegerValue (512),
                   MakeUintegerAccessor (&PoissonGeneratorApplication::m_pktSizeMean),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("Remote", "The address of the destination",
                   AddressValue (),
                   MakeAddressAccessor (&PoissonGeneratorApplication::m_peer),
                   MakeAddressChecker ())
    .AddAttribute ("MaxBytes", 
                   "The total number of bytes to send. Once these bytes are sent, "
                   "no packet is sent again. The value zero means "
                   "that there is no limit.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&PoissonGeneratorApplication::m_maxBytes),
                   MakeUintegerChecker<uint64_t> ())
    .AddAttribute ("Protocol", "The type of protocol to use. This should be "
                   "a subclass of ns3::SocketFactory",
                   TypeIdValue (UdpSocketFactory::GetTypeId ()),
                   MakeTypeIdAccessor (&PoissonGeneratorApplication::m_tid),
                   // This should check for SocketFactory as a parent
                   MakeTypeIdChecker ())
    .AddTraceSource ("Tx", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&PoissonGeneratorApplication::m_txTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("TxWithAddresses", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&PoissonGeneratorApplication::m_txTraceWithAddresses),
                     "ns3::Packet::TwoAddressTracedCallback")
  ;
  return tid;
}


PoissonGeneratorApplication::PoissonGeneratorApplication ()
  : m_socket (0),
    m_connected (false),
    m_lastStartTime (Seconds (0)),
    m_totBytes (0)
{
  NS_LOG_FUNCTION (this);
}

PoissonGeneratorApplication::~PoissonGeneratorApplication()
{
  NS_LOG_FUNCTION (this);
}

void 
PoissonGeneratorApplication::SetMaxBytes (uint64_t maxBytes)
{
  NS_LOG_FUNCTION (this << maxBytes);
  m_maxBytes = maxBytes;
}

Ptr<Socket>
PoissonGeneratorApplication::GetSocket (void) const
{
  NS_LOG_FUNCTION (this);
  return m_socket;
}

void
PoissonGeneratorApplication::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  m_socket = 0;
  // chain up
  Application::DoDispose ();
}

// Application Methods
void PoissonGeneratorApplication::StartApplication () // Called at time specified by Start
{
  NS_LOG_FUNCTION (this);

  // Create the socket if not already
  if (!m_socket)
    {
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
        MakeCallback (&PoissonGeneratorApplication::ConnectionSucceeded, this),
        MakeCallback (&PoissonGeneratorApplication::ConnectionFailed, this));
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

void PoissonGeneratorApplication::StopApplication () // Called at time specified by Stop
{
  NS_LOG_FUNCTION (this);

  CancelEvents ();
  if(m_socket != 0)
    {
      m_socket->Close ();
    }
  else
    {
      NS_LOG_WARN ("PoissonGeneratorApplication found null socket to close in StopApplication");
    }
}

void PoissonGeneratorApplication::CancelEvents ()
{
  NS_LOG_FUNCTION (this);
  m_avgRateFailSafe = m_avgRate;
  Simulator::Cancel (m_sendEvent);
}

// Event handlers
void PoissonGeneratorApplication::StartSending ()
{
  NS_LOG_FUNCTION (this);
  m_lastStartTime = Simulator::Now ();
  ScheduleNextTx ();  // Schedule the send packet event
}

// Private helpers
void PoissonGeneratorApplication::ScheduleNextTx ()
{
  NS_LOG_FUNCTION (this);

  if (m_maxBytes == 0 || m_totBytes < m_maxBytes)
    {
      NS_LOG_UNCOND("Avg Packet Size: "<< (uint64_t) m_pktSizeMean);
      NS_LOG_UNCOND("Avg Data Rate:     "<< (uint64_t) m_avgRate.GetBitRate ());
      
      Ptr<ExponentialRandomVariable> ev_size = CreateObject<ExponentialRandomVariable> ();
      ev_size->SetAttribute ("Mean", DoubleValue (m_pktSizeMean));
      ev_size->SetAttribute("Bound", DoubleValue(1450.0));
      m_pktSize = (uint32_t) ev_size->GetValue();
      NS_LOG_UNCOND("Packet Size: "<<m_pktSize);
      uint32_t bits = m_pktSize * 8;
      //NS_LOG_UNCOND ("bits = " << bits);
      Ptr<ExponentialRandomVariable> ev_delay = CreateObject<ExponentialRandomVariable> ();
      ev_delay->SetAttribute ("Mean", DoubleValue (bits /
                              static_cast<double>(m_avgRate.GetBitRate ())));
      //ev->SetAttribute ("Bound", DoubleValue (totalTime)); 
      double delay = ev_delay->GetValue(); // bits/ static_cast<double>(m_avgRate.GetBitRate ());
      NS_LOG_UNCOND("DELAY:     "<<delay);
      Time nextTime (Seconds (delay)); // Time till next packet
      NS_LOG_LOGIC ("nextTime = " << nextTime);
      m_sendEvent = Simulator::Schedule (nextTime,
                                         &PoissonGeneratorApplication::SendPacket, this);
    }
  else
    { // All done, cancel any pending events
      StopApplication ();
    }
}

void PoissonGeneratorApplication::SendPacket ()
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT (m_sendEvent.IsExpired ());
  Ptr<Packet> packet = Create<Packet> (m_pktSize);
  m_txTrace (packet);
  m_socket->Send (packet);
  m_totBytes += m_pktSize;
  Address localAddress;
  m_socket->GetSockName (localAddress);
  //NS_LOG_UNCOND("BIT RATE-------------------------------"<<m_avgRate<<"    "<<InetSocketAddress::ConvertFrom(m_peer).GetIpv4 ());
  if (InetSocketAddress::IsMatchingType (m_peer))
    {
      NS_LOG_INFO ("At time " << Simulator::Now ().GetSeconds ()
                   << "s on-off application sent "
                   <<  packet->GetSize () << " bytes to "
                   << InetSocketAddress::ConvertFrom(m_peer).GetIpv4 ()
                   << " port " << InetSocketAddress::ConvertFrom (m_peer).GetPort ()
                   << " total Tx " << m_totBytes << " bytes");
      m_txTraceWithAddresses (packet, localAddress, InetSocketAddress::ConvertFrom (m_peer));
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


void PoissonGeneratorApplication::ConnectionSucceeded (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  m_connected = true;
}

void PoissonGeneratorApplication::ConnectionFailed (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
}


} // Namespace ns3