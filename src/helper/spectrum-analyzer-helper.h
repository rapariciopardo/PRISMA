/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef SPECTRUM_ANALYZER_HELPER_H
#define SPECTRUM_ANALYZER_HELPER_H


#include <string>
#include <ns3/attribute.h>
#include <ns3/object-factory.h>
#include <ns3/node-container.h>
#include <ns3/net-device-container.h>
#include <ns3/queue.h>


namespace ns3 {


class SpectrumValue;
class SpectrumChannel;
class SpectrumModel;

class SpectrumAnalyzerHelper
{
public:
  SpectrumAnalyzerHelper ();
  ~SpectrumAnalyzerHelper ();

  /**
   * set the SpectrumChannel that will be used by SpectrumPhy instances created by this helper
   *
   * @param channel
   */
  void SetChannel (Ptr<SpectrumChannel> channel);

  /**
   * set the SpectrumChannel that will be used by SpectrumPhy instances created by this helper
   *
   * @param channel
   */
  void SetChannel (std::string channelName);

  /**
   * @param name the name of the attribute to set
   * @param v the value of the attribute
   *
   * Set these attributes on each HdOfdmSpectrumPhy instance to be created
   */
  void SetPhyAttribute (std::string name, const AttributeValue &v);

  /**
   * @param n1 the name of the attribute to set
   * @param v1 the value of the attribute to set
   *
   * Set these attributes on each AlohaNoackNetDevice created
   */
  void SetDeviceAttribute (std::string n1, const AttributeValue &v1);

  /**
   * Set the spectrum model used by the created SpectrumAnalyzer instances to represent incoming signals
   *
   * @param m 
   */
  void SetRxSpectrumModel (Ptr<SpectrumModel> m);


  /** 
   * Enable ASCII output. This will create one filename for every created SpectrumAnalyzer instance.
   * 
   * @param prefix the prefix of the filename of the traces that will be created.
   */
  void EnableAsciiAll (std::string prefix);

  /**
    * @param c the set of nodes on which a device must be created
    * @return a device container which contains all the devices created by this method.
    */
  NetDeviceContainer Install (NodeContainer c) const;
  /**
   * @param node the node on which a device must be created
   * \returns a device container which contains all the devices created by this method.
   */
  NetDeviceContainer Install (Ptr<Node> node) const;
  /**
   * @param nodeName the name of node on which a device must be created
   * @return a device container which contains all the devices created by this method.
   */
  NetDeviceContainer Install (std::string nodeName) const;


private:
  ObjectFactory m_phy;
  ObjectFactory m_device;
  Ptr<SpectrumChannel> m_channel;
  Ptr<SpectrumModel> m_rxSpectrumModel;
  std::string m_prefix;
};


} // namespace ns3


#endif /* SPECTRUM_ANALYZER_HELPER_H */
