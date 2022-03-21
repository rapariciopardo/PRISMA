/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#ifndef POISSON_GENERATOR_APPLICATION_H
#define POISSON_GENERATOR_APPLICATION_H

#include "ns3/address.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/data-rate.h"
#include "ns3/traced-callback.h"

namespace ns3 {

class Address;
class RandomVariableStream;
class Socket;

/**
 * \ingroup applications 
 * \defgroup onoff PoissonGeneratorApplication
 *
 * This traffic generator follows a Poisson model: after Application::StartApplication
 * is called, packets are randomly generated. The packet size is a random variable
 * that follows an exponential distribution with average "m_avgRate." The inter-arrival
 * time (time between the sending of two consecutive packets) also follows an 
 * exponential distribution with average "m_pktSizeMean."
 */
/**
* If the underlying socket type supports broadcast, this application
* will automatically enable the SetAllowBroadcast(true) socket option.
*/
class PoissonGeneratorApplication : public Application 
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  PoissonGeneratorApplication ();

  virtual ~PoissonGeneratorApplication();

  /**
   * \brief Set the total number of bytes to send.
   *
   * Once these bytes are sent, no packet is sent again.
   * The value zero means that there is no limit.
   *
   * \param maxBytes the total number of bytes to send
   */
  void SetMaxBytes (uint64_t maxBytes);

  /**
   * \brief Return a pointer to associated socket.
   * \return pointer to associated socket
   */
  Ptr<Socket> GetSocket (void) const;

protected:
  virtual void DoDispose (void);
private:
  // inherited from Application base class.
  virtual void StartApplication (void);    // Called at time specified by Start
  virtual void StopApplication (void);     // Called at time specified by Stop

  //helpers
  /**
   * \brief Cancel all pending events.
   */
  void CancelEvents ();

  // Event handlers
  /**
   * \brief Start sending packets
   */
  void StartSending ();
  /**
   * \brief Send a packet
   */
  void SendPacket ();

  Ptr<Socket>     m_socket;       //!< Associated socket
  Address         m_peer;         //!< Peer address
  bool            m_connected;    //!< True if connected
  DataRate        m_avgRate;      //!< Rate that data is generated
  DataRate        m_avgRateFailSafe;      //!< Rate that data is generated (check copy)
  uint32_t        m_pktSize;      //!< Size of packets
  uint32_t        m_pktSizeMean;
  Time            m_lastStartTime; //!< Time last packet sent
  uint64_t        m_maxBytes;     //!< Limit total number of bytes sent
  uint64_t        m_totBytes;     //!< Total bytes sent so far
  EventId         m_sendEvent;    //!< Event id of pending "send packet" event
  TypeId          m_tid;          //!< Type of the socket used

  /// Traced Callback: transmitted packets.
  TracedCallback<Ptr<const Packet> > m_txTrace;

  /// Callbacks for tracing the packet Tx events, includes source and destination addresses
  TracedCallback<Ptr<const Packet>, const Address &, const Address &> m_txTraceWithAddresses;

private:
  /**
   * \brief Schedule the next packet transmission
   */
  void ScheduleNextTx ();
  /**
   * \brief Handle a Connection Succeed event
   * \param socket the connected socket
   */
  void ConnectionSucceeded (Ptr<Socket> socket);
  /**
   * \brief Handle a Connection Failed event
   * \param socket the not connected socket
   */
  void ConnectionFailed (Ptr<Socket> socket);
};

} // namespace ns3

#endif /* POISSON_GENERATOR_APPLICATION_H */
