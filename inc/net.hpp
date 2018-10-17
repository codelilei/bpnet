#ifndef BP_NET_HPP_
#define BP_NET_HPP_

#include <vector>

#include "neuron.hpp"
#include "layer.hpp"

typedef std::vector<float> VF;


class Net
{
 public:
    Net() {}
    Net(float f_lr, int n_in, int n_hid, int n_out) {
        lr = f_lr;
        num_in = n_in;
        num_hid = n_hid;
        num_out = n_out;

        hid_layer = Layer(num_hid);
        hid_layer.InitWeights(num_in);
        hid_layer.ResetBatchPd(num_in);

        out_layer = Layer(num_out);
        out_layer.InitWeights(num_hid);
        out_layer.ResetBatchPd(num_hid);
    }

    VF Forward(VF& in);
    // back propagation for one sample per time
    void BackPropagation(VF& label);
    void UpdateMiniBatch(std::vector<VF>& batch_x, std::vector<VF>& batch_y);
    float CalcErrOnce(VF& input, VF& label);
    float CalcErrTotal(std::vector<VF> &input_x, std::vector<VF> &input_y);

    float TrainNet(std::vector<VF> &inputs, std::vector<VF> &labels, int batch_size,
                   float loss_critera = 0.002, int max_epoch = 10000);

 private:
    float lr;
    int num_in;
    int num_hid;
    int num_out;
    Layer hid_layer;
    Layer out_layer;
};

#endif  // BP_NET_HPP_
