#pragma once
#include <string>

namespace owen2 {
class Position;
}

namespace owen2::sf_nnue {

bool load_sf_net(const std::string& path);
bool is_sf_net_loaded();
int evaluate(const owen2::Position& pos);

}
