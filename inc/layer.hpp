#ifndef BP_LAYER_HPP_
#define BP_LAYER_HPP_

#include <vector>

#include "neuron.hpp"

typedef std::vector<float> VF;


class Layer
{
 public:
    int neuron_num;
    std::vector<Neuron> neurons;
    VF input;

    Layer() {}
    Layer(int num) {
        neuron_num = num;
        for (int i = 0; i < neuron_num; ++i) {
            Neuron neuron;
            neurons.push_back(neuron);
        }
    }

    Layer& operator=(const Layer& layer) {
        if (this == &layer)
            return *this;

        this->neuron_num = layer.neuron_num;
        this->neurons.assign(layer.neurons.begin(), layer.neurons.end());
        this->input.assign(layer.input.begin(), layer.input.end());

        return *this;
    }

    void InitWeights(int in_size) {
        std::vector<Neuron>::iterator itr = neurons.begin();
        for (; itr != neurons.end(); ++itr)
            itr->InitWeights(in_size);
    }

    void ResetBatchPd(int in_size) {
        std::vector<Neuron>::iterator itr = neurons.begin();
        for (; itr != neurons.end(); ++itr)
            itr->ResetBatchPd(in_size);
    }

    void Forward(VF& in, VF& out);
};

#endif  // BP_LAYER_HPP_
