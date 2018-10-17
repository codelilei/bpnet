#include <vector>
#include "layer.hpp"

using std::vector;


void Layer::Forward(VF& in, VF& out) {
    input.assign(in.begin(), in.end());
    std::vector<Neuron>::iterator itr = neurons.begin();
    for (; itr != neurons.end(); ++itr) {
        itr->CalcOutput(in);
        out.push_back(itr->output);
    }
}
