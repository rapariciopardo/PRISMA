// -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*-
//
// Copyright (c) 2008 University of Washington
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
//

#ifndef IPV4_OSPF_ROUTING_H
#define IPV4_OSPF_ROUTING_H

#include <list>
#include <stdint.h>
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-header.h"
#include "ns3/ptr.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/random-variable-stream.h"
//#include "ns3/ospf-router-interface.h"
#include "conf-loader.h"

#include "ns3/simulator.h"

#include <iostream>
using namespace std;

namespace ns3 {

class Packet;
class NetDevice;
class Ipv4Interface;
class Ipv4Address;
class Ipv4Header;
class Ipv4RoutingTableEntry;
class Ipv4MulticastRoutingTableEntry;
class Node;
//class OSPFRouter;

/**
 * \brief OSPF routing protocol for IP version 4 stacks.
 *
 * In ns-3 we have the concept of a pluggable routing protocol.  Routing
 * protocols are added to a list maintained by the Ipv4L3Protocol.  Every 
 * stack gets one routing protocol for free -- the Ipv4StaticRouting routing
 * protocol is added in the constructor of the Ipv4L3Protocol (this is the 
 * piece of code that implements the functionality of the IP layer).
 *
 * As an option to running a dynamic routing protocol, a OSPFRouteManager
 * object has been created to allow users to build routes for all participating
 * nodes.  One can think of this object as a "routing oracle"; it has
 * an omniscient view of the topology, and can construct shortest path
 * routes between all pairs of nodes.  These routes must be stored 
 * somewhere in the node, so therefore this class Ipv4OSPFRouting
 * is used as one of the pluggable routing protocols.  It is kept distinct
 * from Ipv4StaticRouting because these routes may be dynamically cleared
 * and rebuilt in the middle of the simulation, while manually entered
 * routes into the Ipv4StaticRouting may need to be kept distinct.
 *
 * This class deals with Ipv4 unicast routes only.
 *
 * \see Ipv4RoutingProtocol
 * \see OSPFRouteManager
 */
class Ipv4OSPFRouting
{
public:
  static TypeId GetTypeId (void);
/**
 * \brief Construct an empty Ipv4OSPFRouting routing protocol,
 *
 * The Ipv4OSPFRouting class supports host and network unicast routes.
 * This method initializes the lists containing these routes to empty.
 *
 * \see Ipv4OSPFRouting
 */
  Ipv4OSPFRouting ();
  
  ~Ipv4OSPFRouting ();

  // These methods inherited from base class
  //virtual Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p, const Ipv4Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr);
  
  //virtual bool RouteInput  (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
  //                          UnicastForwardCallback ucb, MulticastForwardCallback mcb,
  //                          LocalDeliverCallback lcb, ErrorCallback ecb);
  void NotifyInterfaceUp (uint32_t interface);
  void NotifyInterfaceDown (uint32_t interface);
  void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address);
  void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address);
  void SetIpv4 (Ptr<Ipv4> ipv4);
  void PrintRoutingTable (Ptr<OutputStreamWrapper> stream) const;
  //
  string toString();

  Ptr<Ipv4> getIpv4();

  //Ptr<OSPFRouter> getRouter() const;
  //void setRouter(Ptr<OSPFRouter> router);
  int getID(){return m_id;};
  void setID(int index){m_id = index;};

  int getNNeighbors(){return m_neighbors;};
  void setNNeighbors(int neighbors){m_neighbors = neighbors;};

  void SetOSPFRoutingTable(map<Subnet, vector<int> >& grid);
  map<Subnet, vector<int> >& GetOSPFRoutingTable (void);
  bool update();
  
  bool getUpdateState(){
    return m_update_state;
  };
  void setUpdateState(){m_update_state = true;};
  void resetUpdateState(){m_update_state = false;};

  void sendHelloAll();
  void sendHelloMessage(uint16_t interface);

  void addToNeighbors(int neighbor, Time time);
  map<int, Time>& getNeighbors();
  void removeFromNeighbors(int neighbor);

  void addToLinkStateDatabase(int node, int cost);
  map<int, vector<int> >& getLinkStateDatabase();
  void removeFromLinkStateDatabase(int node);

  void checkNeighbors();
  void sendLSAMessage(int node, int index);

  void handleMessage(Ptr<const Packet> packet);
  void toString(vector<uint16_t>& v);
  void Dijkstra();

  void addNeighbor(int node);
  void removeNeighbor(int node);
  void updateNeighbors();

  void setLSA(map<int, vector<uint16_t> > lsas){
      m_LSAs = lsas;
      Dijkstra();
      cout << toString() << endl;
  };

protected:
  void DoDispose (void);

private:
  vector<int> addNode(int tmp, vector< vector<int> > data);
  void CheckTxQueue();

  static void test();

  void sendMessage(Ipv4Address ip, Ptr<Packet> packet);
  void send2Peer(Ptr<Packet> packet);
  bool m_update_state;

  map<Subnet, vector<int> > m_OSPFRoutingTable;

  map<int, Time> m_LastNeighbors;
  map<int, Time> m_CurNeighbors;

  map<int, vector<uint16_t> > m_LSAs;

  map<int, vector<int> > m_LinkStateDatabase;

  int m_id;
  int m_neighbors;
  /// Set to true if packets are randomly routed among ECMP; set to false for using only one route consistently
  bool m_randomEcmpRouting;
  /// Set to true if this interface should respond to interface events by globallly recomputing routes 
  bool m_respondToInterfaceEvents;
  /// A uniform random number generator for randomly routing packets among ECMP 
  Ptr<UniformRandomVariable> m_rand;

  //Ptr<Ipv4Route> LookupOSPF (Ipv4Address dest, Ptr<NetDevice> oif = 0);

  Ptr<Ipv4Route> LookupOSPFRoutingTable (Ipv4Address source, Ipv4Address dest);

  Ptr<Ipv4> m_ipv4;

};

} // Namespace ns3

#endif /* IPV4_OSPF_ROUTING_H */
