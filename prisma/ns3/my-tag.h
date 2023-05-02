#ifndef MYTAG_H
#define MYTAG_H
#include "ns3/tag.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
namespace ns3 {

class MyTag : public Tag
{
public:
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

  // these are our accessors to our tag structure
  void SetSimpleValue (uint8_t value);
  uint8_t GetSimpleValue (void) const;
  void SetIdValue (uint64_t value);
  uint64_t GetIdValue (void) const;
  void SetSegIndex (uint32_t value);
  uint32_t GetSegIndex (void) const;
  void SetNNIndex (uint32_t value);
  uint32_t GetNNIndex (void) const;
  void SetNodeId (uint32_t value);
  uint32_t GetNodeId (void) const;
  void SetStartTime (uint64_t value);
  uint64_t GetStartTime (void) const;
  void SetFinalDestination (uint32_t value);
  uint32_t GetFinalDestination (void) const;
  void SetLastHop (uint32_t value);
  uint32_t GetLastHop (void) const;
  void SetTrafficValable (uint8_t value);
  uint8_t GetTrafficValable (void) const;
  void SetOverlayIndex (uint32_t value);
  uint32_t GetOverlayIndex (void) const;
  void SetRejectedPacket (uint8_t value);
  uint8_t GetRejectedPacket (void) const;
  void SetNextHop (uint32_t value);
  uint32_t GetNextHop (void) const;
  void SetSource (uint32_t value);
  uint32_t GetSource (void) const;
  void SetTunnelOverlaySendingIndex (uint32_t value);
  uint32_t GetTunnelOverlaySendingIndex (void) const;
  void SetOneHopDelay (float value);
  float GetOneHopDelay (void) const;
private:
  uint8_t m_simpleValue;
  uint64_t m_startTime;
  uint8_t m_trafficValable;
  uint8_t m_rejectedPacket;
  uint32_t m_finalDestination;
  uint32_t m_source;
  uint32_t m_lastHop;
  uint32_t m_nextHop;
  uint64_t m_pktId;
  uint32_t m_segIndex;
  uint32_t m_NNIndex;
  float m_oneHopDelay;
  uint32_t m_nodeId;
  uint32_t m_overlayIndex;
  uint32_t m_tunnelOverlaySendingIndex;  
};
}

#endif