
#include "subnet.h"

#include <cmath>
#include <sstream>
#include <bitset>

namespace ns3{

Subnet::Subnet(){
}

Subnet::Subnet(uint32_t address, int mask){
	uint32_t mask_tmp = 0;
	if (mask>=0 && mask < 32){
		for(int i=31;i>=32-mask;i--){
			mask_tmp += (uint32_t)pow(2, i);
			//cout << mask_tmp << endl;
		}
	}else{
		cout << "invaild parameters" << endl;
		return;
	}
	if((address&(~mask_tmp))!=0){
		cout << "invaild parameters" << endl;
		return;
	}
	this->address = address;
	this->mask = mask_tmp;
}

Subnet::Subnet(string addr, int mask){
	uint32_t address;
	stringstream ss(addr);
	string item;
	while(getline(ss,item,'.')){
		address = address*256+atoi(item.c_str());
	}

	uint32_t mask_tmp = 0;
	if (mask>=0 && mask < 32){
		for(int i=31;i>=32-mask;i--){
			mask_tmp += (uint32_t)pow(2, i);
			//cout << mask_tmp << endl;
		}
	}else{
		cout << "invaild parameters" << endl;
		return;
	}
	if((address&(~mask_tmp))!=0){
		cout << "invaild parameters" << endl;
		return;
	}
	this->address = address;
	this->mask = mask_tmp;
}

Subnet::Subnet(const Subnet& subnet){
	address = subnet.address;
	mask = subnet.mask;
}

Subnet& Subnet::operator=(const Subnet &subnet){
	address = subnet.address;
	mask = subnet.mask;
	return *this;
}

void Subnet::setAddress(uint32_t address){
	this->address = address;
}

void Subnet::setMask(uint32_t mask){
	this->mask = mask;
}

Subnet Subnet::nextSubnet() const{
	Subnet subnet;
	subnet.setAddress(this->address + (~this->mask+1));
	subnet.setMask(this->mask);
	return subnet;
}

Subnet Subnet::prevSubnet() const{
	Subnet subnet;
	subnet.setAddress(this->address - (~this->mask+1));
	subnet.setMask(this->mask);
	return subnet;
}

string Subnet::uint32ToAddress(uint32_t x) const{
	int a = x>>24;
	int b = (x<<8)>>24;
	int c = (x<<16)>>24;
	int d = (x<<24)>>24;
	stringstream ss;
	ss << a << "." << b << "." << c << "." << d;
	return ss.str();
}


string Subnet::toString() const{
	return uint32ToAddress(this->address)+"/"+uint32ToAddress(this->mask);
}

uint32_t Subnet::getAddress() const{
	return this->address;
}

uint32_t Subnet::getMask() const{
	return this->mask;
}

bool Subnet::equals(Subnet other) const{
	if(other.getMask()==this->mask && other.getAddress()==this->address){
		return true;
	}
	return false;
}

bool Subnet::contains(Ipv4Address dest) const{
	return this->address == (this->mask & dest.Get());
}

}