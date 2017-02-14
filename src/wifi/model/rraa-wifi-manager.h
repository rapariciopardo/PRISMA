/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Author: Federico Maguolo <maguolof@dei.unipd.it>
 */

#ifndef RRAA_WIFI_MANAGER_H
#define RRAA_WIFI_MANAGER_H

#include "ns3/traced-value.h"
#include "wifi-remote-station-manager.h"

namespace ns3 {

struct RraaWifiRemoteStation;

/**
 * \brief Robust Rate Adaptation Algorithm
 * \ingroup wifi
 *
 * This is an implementation of RRAA as described in
 * "Robust rate adaptation for 802.11 wireless networks"
 * by "Starsky H. Y. Wong", "Hao Yang", "Songwu Lu", and,
 * "Vaduvur Bharghavan" published in Mobicom 06.
 *
 * This RAA does not support HT, VHT nor HE modes and will error
 * exit if the user tries to configure this RAA with a Wi-Fi MAC
 * that has VhtSupported, HtSupported or HeSupported set.
 */
class RraaWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  RraaWifiManager ();
  virtual ~RraaWifiManager ();

  // Inherited from WifiRemoteStationManager
  void SetHtSupported (bool enable);
  void SetVhtSupported (bool enable);
  void SetHeSupported (bool enable);


private:
  /// ThresholdsItem structure
  struct ThresholdsItem
  {
    uint32_t datarate; ///< data rate
    double pori; ///< PORI
    double pmtl; ///< PMTL
    uint32_t ewnd; ///< EWND
  };

  //overriden from base class
  WifiRemoteStation * DoCreateStation (void) const;
  void DoReportRxOk (WifiRemoteStation *station,
                     double rxSnr, WifiMode txMode);
  void DoReportRtsFailed (WifiRemoteStation *station);
  void DoReportDataFailed (WifiRemoteStation *station);
  void DoReportRtsOk (WifiRemoteStation *station,
                      double ctsSnr, WifiMode ctsMode, double rtsSnr);
  void DoReportDataOk (WifiRemoteStation *station,
                       double ackSnr, WifiMode ackMode, double dataSnr);
  void DoReportFinalRtsFailed (WifiRemoteStation *station);
  void DoReportFinalDataFailed (WifiRemoteStation *station);
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station);
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station);
  bool DoNeedRts (WifiRemoteStation *st,
                  Ptr<const Packet> packet, bool normally);
  bool IsLowLatency (void) const;

  /**
   * Return the index for the maximum transmission rate for
   * the given station.
   *
   * \param station
   *
   * \return the index for the maximum transmission rate
   */
  uint32_t GetMaxRate (RraaWifiRemoteStation *station);
  /**
   * Return the index for the minimum transmission rate for
   * the given station.
   *
   * \param station
   *
   * \return the index for the minimum transmission rate
   */
  uint32_t GetMinRate (RraaWifiRemoteStation *station);
  /**
   * Check if the counter should be resetted.
   *
   * \param station
   */
  void CheckTimeout (RraaWifiRemoteStation *station);
  /**
   * Find an appropriate rate for the given station, using
   * a basic algorithm.
   *
   * \param station
   */
  void RunBasicAlgorithm (RraaWifiRemoteStation *station);
  /**
   * Activate the use of RTS for the given station if the conditions are met.
   *
   * \param station
   */
  void ARts (RraaWifiRemoteStation *station);
  /**
   * Reset the counters of the given station.
   *
   * \param station
   */
  void ResetCountersBasic (RraaWifiRemoteStation *station);
  /**
   * Get a threshold for the given mode.
   *
   * \param mode
   * \param station
   *
   * \return threshold
   */
  ThresholdsItem GetThresholds (WifiMode mode, RraaWifiRemoteStation *station) const;
  /**
   * Get a threshold for the given station and mode index.
   *
   * \param station
   * \param rate
   *
   * \return threshold
   */
  ThresholdsItem GetThresholds (RraaWifiRemoteStation *station, uint32_t rate) const;

  bool m_basic; ///< basic
  Time m_timeout; ///< timeout
  uint32_t m_ewndfor54; ///< ewndfor54
  uint32_t m_ewndfor48; ///< ewndfor48
  uint32_t m_ewndfor36; ///< ewndfor36
  uint32_t m_ewndfor24; ///< ewndfor24
  uint32_t m_ewndfor18; ///< ewndfor18
  uint32_t m_ewndfor12; ///< ewndfor12
  uint32_t m_ewndfor9; ///< ewndfor9
  uint32_t m_ewndfor6; ///< ewndfor6
  double m_porifor48; ///< porifor48
  double m_porifor36; ///< porifor36
  double m_porifor24; ///< porifor24
  double m_porifor18; ///< porifor18
  double m_porifor12; ///< porifor12
  double m_porifor9; ///< porifor9
  double m_porifor6; ///< porifor6
  double m_pmtlfor54; ///< pmtlfor54
  double m_pmtlfor48; ///< pmtlfor48
  double m_pmtlfor36; ///< pmtlfor36
  double m_pmtlfor24; ///< pmtlfor24
  double m_pmtlfor18; ///< pmtlfor18
  double m_pmtlfor12; ///< pmtlfor12
  double m_pmtlfor9; ///< pmtlfor9

  TracedValue<uint64_t> m_currentRate; //!< Trace rate changes
};

} //namespace ns3

#endif /* RRAA_WIFI_MANAGER_H */
