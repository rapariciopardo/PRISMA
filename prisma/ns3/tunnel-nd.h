#ifndef TUNNEL_ND_H
#define TUNNEL_ND_H

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
 
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/virtual-net-device.h"

using namespace std;

namespace ns3{

class TunnelNetDevice
{
  Ptr<Socket> m_Socket; 
  Ipv4Address m_Address; 

  Ptr<UniformRandomVariable> m_rng; 
  Ptr<VirtualNetDevice> m_Tap;  
 
  bool NVirtualSend (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  void NSocketRecv (Ptr<Socket> socket);
 
public:
 
  TunnelNetDevice (Ptr<Node> n, Ipv4Address nAddr);
 
 
};
 

}
#endif
