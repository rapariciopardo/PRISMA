#ifndef OSPF_TAG_H
#define OSPF_TAG_H

#include "ns3/tag.h"
#include "ns3/uinteger.h"
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

namespace ns3{

class OSPFTag : public Tag
{
public:
   OSPFTag(){this->type = 0;};
   static TypeId GetTypeId (void);
   virtual TypeId GetInstanceTypeId (void) const;
   virtual uint32_t GetSerializedSize (void) const;
   virtual void Serialize (TagBuffer i) const;
   virtual void Deserialize (TagBuffer i);
   virtual void Print (std::ostream &os) const;

   // these are our accessors to our tag structure

   void setType(uint8_t type){this->type = type;};
   uint8_t getType(){return this->type;};

   void setNode(uint16_t node){this->node = node;};
   uint16_t getNode(){return this->node;};  

   void setLSA(int node, vector<uint16_t>& lsa){
      this->lsa_node = node;
      this->lsa_size = lsa.size();
      this->lsa = lsa;
   };

   uint16_t getLSANode(){return lsa_node;};
   uint16_t getLSASize(){return lsa_size;};
   vector<uint16_t>& getLSA(){return lsa;};

   void setLSA(int node, uint32_t index){
      this->lsa_node = node;
      this->index = index;
   };

   uint32_t getLSAIndex(){
      return this->index;
   };

private:
   //1 -- hello
   //2 -- LSA update
   uint8_t type;
   uint16_t node;

   uint16_t lsa_node;
   uint16_t lsa_size;
   vector<uint16_t> lsa;
   uint32_t index;
};

}
#endif
