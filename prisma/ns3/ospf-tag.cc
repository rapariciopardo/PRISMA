#include "ospf-tag.h"
#include "ns3/log.h"

using namespace ns3; 

NS_LOG_COMPONENT_DEFINE ("OSPFTag");

NS_OBJECT_ENSURE_REGISTERED (OSPFTag);


TypeId OSPFTag::GetTypeId (void){
   static TypeId tid = TypeId ("ns3::OSPFTag")
                   .SetParent<Tag> ()
                   .AddConstructor<OSPFTag> ();
           return tid;
}

TypeId OSPFTag::GetInstanceTypeId (void) const{
    return GetTypeId ();
}

uint32_t OSPFTag::GetSerializedSize (void) const{
	uint32_t size = 0;
	size = 3;
	if(type == 2){
		size = size + 2 + 4;
	}
    return size;
}

void OSPFTag::Serialize (TagBuffer i) const{    
  i.WriteU8(type);
  i.WriteU16(node);
  if(type == 2){
  		i.WriteU16(lsa_node);
      i.WriteU32(index);
  		/*i.WriteU16(lsa_size);
  		for(int k=0;k<(int)lsa_size;k++){
  			i.WriteU16(lsa[k]);
  		}*/
  }
}

void OSPFTag::Deserialize (TagBuffer i){
    type = i.ReadU8();
    node = i.ReadU16();
    if(type == 2){
    	lsa_node = i.ReadU16();
      index = i.ReadU32();
    	/*lsa_size = i.ReadU16();
    	lsa.clear();
    	for(int k=0;k<(int)lsa_size;k++){
  			lsa.push_back(i.ReadU16());
  		}*/
    }
}

void OSPFTag::Print (std::ostream &os) const{
    os << endl;
}
