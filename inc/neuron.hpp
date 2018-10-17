#ifndef BP_NEURON_HPP_
#define BP_NEURON_HPP_

#include <vector>

typedef std::vector<float> VF;


class Neuron
{
public:
    float bias;
    VF weights;
    float batch_pdb;
    VF batch_pdw;
    float output;

    Neuron() {
        bias = GetRandom(-1, 1);
        output = 0;
        batch_pdb = 0;
    }

    float GetRandom(float min, float max) {
        return ((float)rand() * (max - min) / (float)RAND_MAX + min);
    }

    void InitWeights(int in_size) {
        for (int i = 0; i < in_size; ++i)
            weights.push_back(GetRandom(-1, 1));
    }

    void ResetBatchPd(int in_size) {
        batch_pdb = 0;
        batch_pdw.assign(in_size, 0);
    }

    void CalcOutput(VF& input);
    float GetDSigMoid();
};

#endif  // BP_NEURON_HPP_
