#include "neuron.hpp"


void Neuron::CalcOutput(VF& input) {
    float sum = 0;
    for (int i = 0; i < input.size(); ++i)
        sum += input[i] * weights[i] + bias;

    //activation function
    //output = 1 / (1 + exp(-sum));      //sigmoid 
    output = 2 / (1 + exp(-sum)) - 1;    //bipolar sigmoid 
    //output = tanh(sum);                //hyperbolic tangent
}

float Neuron::GetDSigMoid() {
    //return output * (1 - output);      //for sigmoid
    return (1 - output * output) / 2;    //for tanh
}
