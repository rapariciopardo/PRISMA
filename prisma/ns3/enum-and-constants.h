#ifndef ENUMS_AND_CONSTANTS
#define ENUMS_AND_CONSTANTS
namespace ns3{

enum PacketType{
    DATA_PACKET = 0,
    BIG_SIGN_PACKET = 1,
    SMALL_SIGN_PACKET = 2,
    PING_FORWARD_PACKET= 3,
    PING_BACK_PACKET = 4,
};
}

#endif // ENUMS_AND_CONTANTS