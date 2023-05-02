#include <fstream>
#include <iostream>

#include "conf-loader.h"
//#include "ns3/log.h"


//NS_LOG_COMPONENT_DEFINE ("ConfLoader");

namespace ns3{

using namespace std;

ConfLoader* ConfLoader::m_pInstance = NULL;

ConfLoader* ConfLoader::Instance(){
	if(!m_pInstance){
		m_pInstance = new ConfLoader;
	}
	return m_pInstance;
}

void ConfLoader::setCoreNum(int num){
    m_CoreNum = num;
}
void ConfLoader::setToRNum(int num){
    m_ToRNum = num;
}
void ConfLoader::setBorderNum(int num){
    m_BorderNum = num;
}
void ConfLoader::setSubnetMask(int mask){
    m_SubnetMask = mask;
}
void ConfLoader::setAddressStart(uint32_t address){
    m_AddressStart = address;
}
int ConfLoader::getCoreNum() const{
    return m_CoreNum;
}
int ConfLoader::getToRNum() const{
    return m_ToRNum;
}
int ConfLoader::getBorderNum() const{
    return m_BorderNum;
}
int ConfLoader::getSubnetMask() const{
    return m_SubnetMask;
}
uint32_t ConfLoader::getAddressStart() const{
    return m_AddressStart;
}

int ConfLoader::getTotalNum() const{
    return m_CoreNum+m_ToRNum+m_BorderNum;
}


void ConfLoader::setNodeContainer(NodeContainer& nc){
    m_nodes = nc;
}

NodeContainer& ConfLoader::getNodeContainer(){
    return m_nodes;
}

//----------------------

int ConfLoader::calcDestNodeBySource(int id, int interface){
    if(id < m_CoreNum){
        return m_CoreNum+interface-1;
    }else if(id < getTotalNum()){
        if(interface<=m_CoreNum){
            return interface-1;
        }else{
            return id+m_ToRNum+m_BorderNum;
        }
    }else{
        return id - (m_ToRNum+m_BorderNum);
    }
}

int ConfLoader::calcDestInterfaceBySource(int id, int interface){
    if(id < m_CoreNum){
        return id+1;
    }else if(id < getTotalNum()){
        if(interface<=m_CoreNum){
            return id-m_CoreNum+1;
        }else{
            return 1;
        }
    }else{
        return m_CoreNum+1;
    }
}

//node >= getTotalNum()
// may have bugs
int ConfLoader::calcSourceInterfaceByNode(int id, int node){
    if (id==node) return 0;
    if(id < m_CoreNum){
        return node-getCoreNum()+1;
    }else if(id < getTotalNum()){
        if(node-id == m_ToRNum+m_BorderNum){
            return m_CoreNum+1;
        }else{
            return node+1;
        }
    }else{
        return 1;
    }
}

Subnet& ConfLoader::calcSubnetByNode(int node){
    return m_Subnets[node];
}

map<pair<int,int>, Subnet>& ConfLoader::getLinkSubnetMap(){
    return m_LinkSubnet;
}

void ConfLoader::addItem2LinkSubnetMap(int index1, int index2, Subnet& subnet){
    m_LinkSubnet[std::make_pair(index1,index2)] = subnet;
}
  
pair<int,int> ConfLoader::getLinkBySubnet(Subnet& subnet){
    for(map<pair<int,int>, Subnet>::iterator it = m_LinkSubnet.begin(); it!=m_LinkSubnet.end(); ++it){
        if (it->second.equals(subnet)){
            return it->first;
        }
    }
    return pair<int,int>();
}

Subnet& ConfLoader::getSubnetByID(int index1, int index2){
    return m_LinkSubnet[std::make_pair(index1,index2)];
}

void ConfLoader::update(int id, map<Subnet, int>& table){
  if(id < m_CoreNum){
      for(int i=0;i<m_ToRNum+m_BorderNum;i++){
          table[getSubnetByID(id, i+m_CoreNum)] = i+1;
      }
      for(int i=0;i<m_ToRNum;i++){
          table[getSubnetByID(i+m_CoreNum, i+getTotalNum())] = i+1;
      }
      /*std::vector<Subnet> subnets = ConfLoader::Instance()->getSubnets();
      for(std::vector<Subnet>::iterator it = subnets.begin(); it!=subnets.end(); ++it){
          if(table.find(*it)==table.end()){
          }
      }*/
  }else if(id<getTotalNum()){
      for(int i=0; i<m_CoreNum; i++){
          table[getSubnetByID(i,id)]=i+1;
      }
      for(int i=0; i<m_ToRNum; i++){
        if(id == (i+m_CoreNum)){
            table[getSubnetByID(id, id+m_ToRNum+m_BorderNum)] = m_CoreNum+1;
        }else{
            table[getSubnetByID(i+m_CoreNum, i+m_CoreNum+m_ToRNum+m_BorderNum)] = 1;
        }
      }
  }else{
      for(std::vector<Subnet>::iterator it = m_Subnets.begin(); it!=m_Subnets.end(); ++it){
          table[*it] = 1;
      }
  }
}

}
