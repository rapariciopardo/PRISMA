#ifndef CONF_LOADER_H
#define CONF_LOADER_H

#include <string>
#include <iostream>
#include <stdint.h>
#include <map>

#include "ns3/node-container.h"
#include "subnet.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"

using namespace std;

namespace ns3{

class NodeContainer;
class Ipv4Address;

class ConfLoader
{
public:
  static ConfLoader* Instance();
  void setCoreNum(int num);
  void setToRNum(int num);
  void setBorderNum(int num);
  void setSubnetMask(int mask);
  void setAddressStart(uint32_t address);
  int getCoreNum() const;
  int getToRNum() const;
  int getBorderNum() const;
  int getSubnetMask() const;
  int getTotalNum() const;

  uint32_t getAddressStart() const;


  map<pair<int,int>, Subnet>& getLinkSubnetMap();
  void addItem2LinkSubnetMap(int index1, int index2, Subnet& subnet);
  pair<int,int> getLinkBySubnet(Subnet& subnet);
  Subnet& getSubnetByID(int index1, int index2);

  void setNodeContainer(NodeContainer& nc);
  NodeContainer& getNodeContainer();

  void incrementLossPacketCounter(int id){ 
    if(m_lossPacketCounter.find(id)==m_lossPacketCounter.end()){
        m_lossPacketCounter[id] = 0;
    }
    m_lossPacketCounter[id]++;
  };

  void prepareLinkDown(){
  }

  map<int,int>& getLossPacketCounter(){ return this->m_lossPacketCounter;};

  void setCurrentTime(Time time){
    if(!isDown){ return;}
    m_stopTime = time;
    if(m_startTime.IsZero()){
        m_startTime = time;
    }
  };

  Time getDiffTime(){
      return m_stopTime - m_startTime;
  };

  Time& getStartTime(){
    return m_startTime;
  };

  Time& getStopTime(){ return m_stopTime;};

  int calcDestNodeBySource(int id, int interface);
  int calcDestInterfaceBySource(int id, int interface);
  int calcSourceInterfaceByNode(int id, int node);
  Subnet& calcSubnetByNode(int node);

  vector<Subnet>& getSubnets(){
      return m_Subnets;
  }

  void addSubnet(Subnet& subnet){
      m_Subnets.push_back(subnet);
  }

  void update(int id, map<Subnet, int>& table);

  string getHelloMsgString(){
      return string("hello");
  }

  void setUnavailableInterval(int UnavailableInterval){
      this->m_UnavailableInterval = UnavailableInterval;
  };

  int getUnavailableInterval(){
      return this->m_UnavailableInterval;
  };

  void addToNodeSubnet(int id, Subnet& subnet){
      m_NodeSubnets[id] = subnet;
  };

  Subnet& getSubnetByNode(int id){
      return m_NodeSubnets[id];
  };

  string PrintMap(map<int,int> m){
      stringstream ss;
      for(map<int, int>::iterator it = m.begin(); it!=m.end(); ++it){
          ss << it->first << ":" << it->second << endl;
      }
      return ss.str();
  };

  void addLSA(int index, vector<uint16_t>& lsa){ 
    m_lsas[index] = lsa;
    m_lsaNum ++;
  };

  int getLSANum(){return m_lsaNum;};
  
  vector<uint16_t>& getLSA(int index){ return m_lsas[index];};

  void incrementSuccessPacket(int id){
    if(m_SuccessPacket.find(id)==m_SuccessPacket.end()){
        m_SuccessPacket[id] = 0;
    }
    m_SuccessPacket[id]++;
  };

  map<int,int>& getSuccessPacket(){return m_SuccessPacket;};

  void incrementSendPacket(int id){
    if(m_SendPacket.find(id)==m_SendPacket.end()){
        m_SendPacket[id] = 0;
    }
    m_SendPacket[id]++;
  };

  map<int,int>& getSendPacket(){return m_SendPacket;};

  void incrementRecvPacket(int id){
    if(m_RecvPacket.find(id)==m_RecvPacket.end()){
        m_RecvPacket[id] = 0;
    }
    m_RecvPacket[id]++;
  };

  map<int,int>& getRecvPacket(){return m_RecvPacket;};

  int getNodeByInterface(int id, int interface){
      if(id < m_CoreNum){
          return m_CoreNum+interface-1;
      }else if(id<m_CoreNum+m_ToRNum+m_BorderNum){
          if(interface==m_CoreNum+1) return id+m_ToRNum+m_BorderNum;
          else return interface-1;
      }else{
        return id - (m_ToRNum+m_BorderNum);
      }
      return -1;
  };
  

  uint32_t getPacketReceiveDelay(){return m_PacketReceiveDelay;};
  void setPacketReceiveDelay(uint32_t delay){m_PacketReceiveDelay = delay;};
  void setCongestionWaningLimit(float limit){m_CongestionWarningLimit = limit;};
  float getCongestionWaningLimit(){return m_CongestionWarningLimit;};
  void setCalculateCost(float cost){m_CalculateCost = cost;};
  float getCalculateCost(){return m_CalculateCost;};

  void setLSPDelay(float cost){m_LSPDelay = cost;};
  float getLSPDelay(){return m_LSPDelay;};
private:

	ConfLoader(){
    isDown = false;
    m_lsaNum = 0;

  };
	ConfLoader(ConfLoader const&){};
	//ConfLoader& operator=(ConfLoader const&){};
	static ConfLoader* m_pInstance;

  float m_CongestionWarningLimit; 
  float m_CalculateCost;
  
  float m_LSPDelay;
  enum NodeType{
      CORE = 0,
      TOR,
      BORDER,
  };

  map<pair<int,int>, Subnet> m_LinkSubnet;
  std::vector<Subnet> m_Subnets;
  map<int, Subnet> m_NodeSubnets;

  int m_UnavailableInterval;
  int m_CoreNum;
  int m_ToRNum;
  int m_BorderNum;
  int m_SubnetMask;
  uint32_t m_AddressStart;
  uint32_t m_PacketReceiveDelay;

  bool isDown;
  Time m_startTime;
  Time m_stopTime;

  map<int,int > m_lossPacketCounter;
  map<int,int> m_SuccessPacket;
  map<int,int> m_SendPacket;
  map<int,int> m_RecvPacket;

  NodeContainer m_nodes;

  map<int, vector<uint16_t> > m_lsas;
  int m_lsaNum;
};

}

#endif /* CONF_LOADER_H */