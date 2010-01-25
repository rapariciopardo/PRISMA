/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef CSMA_HELPER_H
#define CSMA_HELPER_H

#include <string>

#include "ns3/attribute.h"
#include "ns3/object-factory.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/csma-channel.h"
#include "ns3/deprecated.h"
#include "pcap-helper.h"
#include "ascii-trace-helper.h"

namespace ns3 {

class Packet;

/**
 * \brief build a set of CsmaNetDevice objects
 */
class CsmaHelper : public PcapUserHelperForDevice, public AsciiTraceUserHelperForDevice
{
public:
  /**
   * Construct a CsmaHelper.
   */
  CsmaHelper ();

  /**
   * \param type the type of queue
   * \param n1 the name of the attribute to set on the queue
   * \param v1 the value of the attribute to set on the queue
   * \param n2 the name of the attribute to set on the queue
   * \param v2 the value of the attribute to set on the queue
   * \param n3 the name of the attribute to set on the queue
   * \param v3 the value of the attribute to set on the queue
   * \param n4 the name of the attribute to set on the queue
   * \param v4 the value of the attribute to set on the queue
   *
   * Set the type of queue to create and associated to each
   * CsmaNetDevice created through CsmaHelper::Install.
   */
  void SetQueue (std::string type,
                 std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                 std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                 std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                 std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue ());

  /**
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   *
   * Set these attributes on each ns3::CsmaNetDevice created
   * by CsmaHelper::Install
   */
  void SetDeviceAttribute (std::string n1, const AttributeValue &v1);

  /**
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   *
   * Set these attributes on each ns3::CsmaChannel created
   * by CsmaHelper::Install
   */
  void SetChannelAttribute (std::string n1, const AttributeValue &v1);

  /**
   * This method creates an ns3::CsmaChannel with the attributes configured by
   * CsmaHelper::SetChannelAttribute, an ns3::CsmaNetDevice with the attributes
   * configured by CsmaHelper::SetDeviceAttribute and then adds the device
   * to the node and attaches the channel to the device.
   *
   * \param node The node to install the device in
   * \returns A containter holding the added net device.
   */
  NetDeviceContainer Install (Ptr<Node> node) const;

  /**
   * This method creates an ns3::CsmaChannel with the attributes configured by
   * CsmaHelper::SetChannelAttribute, an ns3::CsmaNetDevice with the attributes
   * configured by CsmaHelper::SetDeviceAttribute and then adds the device
   * to the node and attaches the channel to the device.
   *
   * \param name The name of the node to install the device in
   * \returns A containter holding the added net device.
   */
  NetDeviceContainer Install (std::string name) const;

  /**
   * This method creates an ns3::CsmaNetDevice with the attributes configured by
   * CsmaHelper::SetDeviceAttribute and then adds the device to the node and 
   * attaches the provided channel to the device.
   *
   * \param node The node to install the device in
   * \param channel The channel to attach to the device.
   * \returns A containter holding the added net device.
   */
  NetDeviceContainer Install (Ptr<Node> node, Ptr<CsmaChannel> channel) const;

  /**
   * This method creates an ns3::CsmaNetDevice with the attributes configured by
   * CsmaHelper::SetDeviceAttribute and then adds the device to the node and 
   * attaches the provided channel to the device.
   *
   * \param node The node to install the device in
   * \param channelName The name of the channel to attach to the device.
   * \returns A containter holding the added net device.
   */
  NetDeviceContainer Install (Ptr<Node> node, std::string channelName) const;

  /**
   * This method creates an ns3::CsmaNetDevice with the attributes configured by
   * CsmaHelper::SetDeviceAttribute and then adds the device to the node and 
   * attaches the provided channel to the device.
   *
   * \param nodeName The name of the node to install the device in
   * \param channel The chanel to attach to the device.
   * \returns A containter holding the added net device.
   */
  NetDeviceContainer Install (std::string nodeName, Ptr<CsmaChannel> channel) const;

  /**
   * This method creates an ns3::CsmaNetDevice with the attributes configured by
   * CsmaHelper::SetDeviceAttribute and then adds the device to the node and 
   * attaches the provided channel to the device.
   *
   * \param nodeName The name of the node to install the device in
   * \param channelName The name of the chanel to attach to the device.
   * \returns A containter holding the added net device.
   */
  NetDeviceContainer Install (std::string nodeName, std::string channelName) const;

  /**
   * This method creates an ns3::CsmaChannel with the attributes configured by
   * CsmaHelper::SetChannelAttribute.  For each Ptr<node> in the provided
   * container: it creates an ns3::CsmaNetDevice (with the attributes 
   * configured by CsmaHelper::SetDeviceAttribute); adds the device to the 
   * node; and attaches the channel to the device.
   *
   * \param c The NodeContainer holding the nodes to be changed.
   * \returns A containter holding the added net devices.
   */
  NetDeviceContainer Install (const NodeContainer &c) const;

  /**
   * For each Ptr<node> in the provided container, this method creates an 
   * ns3::CsmaNetDevice (with the attributes configured by 
   * CsmaHelper::SetDeviceAttribute); adds the device to the node; and attaches 
   * the provided channel to the device.
   *
   * \param c The NodeContainer holding the nodes to be changed.
   * \param channel The channel to attach to the devices.
   * \returns A containter holding the added net devices.
   */
  NetDeviceContainer Install (const NodeContainer &c, Ptr<CsmaChannel> channel) const;

  /**
   * For each Ptr<node> in the provided container, this method creates an 
   * ns3::CsmaNetDevice (with the attributes configured by 
   * CsmaHelper::SetDeviceAttribute); adds the device to the node; and attaches 
   * the provided channel to the device.
   *
   * \param c The NodeContainer holding the nodes to be changed.
   * \param channelName The name of the channel to attach to the devices.
   * \returns A containter holding the added net devices.
   */
  NetDeviceContainer Install (const NodeContainer &c, std::string channelName) const;

private:
  /*
   * \internal
   */
  Ptr<NetDevice> InstallPriv (Ptr<Node> node, Ptr<CsmaChannel> channel) const;

  /**
   * \brief Enable pcap output on the indicated net device.
   * \internal
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * \param prefix Filename prefix to use for pcap files.
   * \param nd Net device for which you want to enable tracing.
   * \param promiscuous If true capture all possible packets available at the device.
   */
  virtual void EnablePcapInternal (std::string prefix, Ptr<NetDevice> nd, bool promiscuous = false);

  /**
   * \brief Enable ascii trace output on the indicated net device.
   * \internal
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * \param stream The output stream object to use when logging ascii traces.
   * \param prefix Filename prefix to use for ascii trace files.
   * \param nd Net device for which you want to enable tracing.
   */
  virtual void EnableAsciiInternal (Ptr<OutputStreamObject> stream, std::string prefix, Ptr<NetDevice> nd);

  ObjectFactory m_queueFactory;
  ObjectFactory m_deviceFactory;
  ObjectFactory m_channelFactory;
};

} // namespace ns3

#endif /* CSMA_HELPER_H */
