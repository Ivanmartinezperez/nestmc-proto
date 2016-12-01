#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <sstream>
#include "json.hpp"
#include "kernels.h"

using json = nlohmann::json;



template<typename T>
void copyContainer(std::vector<T> &start,T* &end ){
    end = (T*) malloc(start.size()*sizeof start[0]);
    std::copy(start.begin(),start.end(),end);
}

/*template<typename T>
void Interleave(T* src,T* dst,int system_size){

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < BlockSize; ++j)
    {
      dst[i*BlockSize+j] = src[j*system_size+i];
    }
  }
}*/


HinesMatrix loadMatrix(std::string filename){


	std::ifstream file(filename.c_str());
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::string str = buffer.str();
    //std::cout << str;

	
  	json j = json::parse(str);

    //std::cout << j["rhs_"] << std::endl;
  	
  	auto d = j["d"].get<std::vector<double>>();
  	auto a = j["l"].get<std::vector<double>>();
  	auto p = j["p"].get<std::vector<int>>();
  	auto rhs = j["rhs_"].get<std::vector<double>>();
  	auto sol = j["sol"].get<std::vector<double>>();
  	auto b = j["u"].get<std::vector<double>>();
  	auto cell_size = j["size"].get<int>();

  	HinesMatrix hs;


  	copyContainer(a,hs.a);
  	copyContainer(b,hs.b);
  	copyContainer(d,hs.d);
    copyContainer(rhs,hs.rhs);
  	copyContainer(p,hs.p);
  	hs.num_cells=1;
  	hs.cell_size=cell_size;

  	file.close();

  	return hs;
}

