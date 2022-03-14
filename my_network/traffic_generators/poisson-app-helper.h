/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#ifndef POISSON_APP_HELPER_H
#define POISSON_APP_HELPER_H

#include <stdint.h>
#include <string>
#include "ns3/object-factory.h"
#include "ns3/address.h"
#include "ns3/attribute.h"
#include "ns3/net-device.h"
#include "ns3/node-container.h"
#include "ns3/application-container.h"
//#include "ns3/onoff-application.h"

namespace ns3 {

class DataRate;

/**
 * \ingroup onoff
 * \brief A helper to make it easier to instantiate an ns3::PoissonGeneratorApplication 
 * on a set of nodes.
 */
class PoissonAppHelper
{
public:
  /**
   * Create an PoissonAppHelper to make it easier to work with PoissonGeneratorApplications
   *
   * \param protocol the name of the protocol to use to send traffic
   *        by the applications. This string identifies the socket
   *        factory type used to create sockets for the applications.
   *        A typical value would be ns3::UdpSocketFactory.
   * \param address the address of the remote node to send traffic
   *        to.
   */
  PoissonAppHelper (std::string protocol, Address address);

  /**
   * Helper function used to set the underlying application attributes.
   *
   * \param name the name of the application attribute to set
   * \param value the value of the application attribute to set
   */
  void SetAttribute (std::string name, const AttributeValue &value);

  /**
   * Helper function to set the average rate source.  
   *
   * \param dataRate: DataRate object for the average sending rate
   * \param packetSize: average size in bytes of the packet payloads generated
   */
  void SetAverageRate (DataRate dataRate, uint32_t packetSize = 512);

  /**
   * Install an ns3::PoissonGeneratorApplication on each node of the input container
   * configured with all the attributes set with SetAttribute.
   *
   * \param c NodeContainer of the set of nodes on which an PoissonGeneratorApplication 
   * will be installed.
   * \returns Container of Ptr to the applications installed.
   */
  ApplicationContainer Install (NodeContainer c) const;

  /**
   * Install an ns3::PoissonGeneratorApplication on the node configured with all the 
   * attributes set with SetAttribute.
   *
   * \param node The node on which an PoissonGeneratorApplication will be installed.
   * \returns Container of Ptr to the applications installed.
   */
  ApplicationContainer Install (Ptr<Node> node) const;

  /**
   * Install an ns3::PoissonGeneratorApplication on the node configured with all the 
   * attributes set with SetAttribute.
   *
   * \param nodeName The node on which an PoissonGeneratorApplication will be installed.
   * \returns Container of Ptr to the applications installed.
   */
  ApplicationContainer Install (std::string nodeName) const;

private:
  /**
   * Install an ns3::PoissonGeneratorApplication on the node configured with all the 
   * attributes set with SetAttribute.
   *
   * \param node The node on which an PoissonGeneratorApplication will be installed.
   * \returns Ptr to the application installed.
   */
  Ptr<Application> InstallPriv (Ptr<Node> node) const;

  ObjectFactory m_factory; //!< Object factory.
};

} // namespace ns3

#endif /* POISSON_APP_HELPER_H */

