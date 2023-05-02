#include "tunnel-nd.h"
#include "ns3/log.h"

using namespace ns3; 

NS_LOG_COMPONENT_DEFINE ("TunnelNetDevice");

bool
TunnelNetDevice::NVirtualSend (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_DEBUG ("Send to " << m_Address << ": " << *packet);
  m_Socket->SendTo (packet, 0, InetSocketAddress (m_Address, 667));
  return true;
}

void 
TunnelNetDevice::NSocketRecv (Ptr<Socket> socket)
{
  Ptr<Packet> packet = socket->Recv (65535, 0);
  NS_LOG_DEBUG ("N0SocketRecv: " << *packet);
  m_Tap->Receive (packet, 0x0800, m_Tap->GetAddress (), m_Tap->GetAddress (), NetDevice::PACKET_HOST);
}

TunnelNetDevice::TunnelNetDevice (Ptr<Node> n, Ipv4Address nAddr) : m_Address (nAddr)
{
  m_rng = CreateObject<UniformRandomVariable> ();
  m_Socket = Socket::CreateSocket (n, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  m_Socket->Bind (InetSocketAddress (Ipv4Address::GetAny (), 667));
  m_Socket->SetRecvCallback (MakeCallback (&TunnelNetDevice::NSocketRecv, this));

  // n0 tap device
  m_Tap = CreateObject<VirtualNetDevice> ();
  m_Tap->SetAddress (Mac48Address ("11:00:01:02:03:01"));
  m_Tap->SetSendCallback (MakeCallback (&TunnelNetDevice::NVirtualSend, this));
  n->AddDevice (m_Tap);
  Ptr<Ipv4> ipv4 = n->GetObject<Ipv4> ();
  uint32_t i = ipv4->AddInterface (m_Tap);
  ipv4->AddAddress (i, Ipv4InterfaceAddress (Ipv4Address ("11.0.0.1"), Ipv4Mask ("255.255.255.0")));
  ipv4->SetUp (i);
}