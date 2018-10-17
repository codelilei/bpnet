#include <iostream>
#include <vector>

#include "data.hpp"
#include "net.hpp"

using std::vector;
using std::cout;
using std::endl;


VF Net::Forward(VF& in) {
    VF hid_out, out;
    hid_layer.Forward(in, hid_out);
    out_layer.Forward(hid_out, out);

    return out;
}

// back propagation for one sample per time
void Net::BackPropagation(VF& label) {
    // calculate delta: partial derivative of error to no activation output
    // last output layer differs with others on calculating delta
    VF deltas_out;
    for (int i = 0; i < out_layer.neuron_num; ++i) {
        float delta_out = (out_layer.neurons[i].output - label[i]) * out_layer.neurons[i].GetDSigMoid();
        deltas_out.push_back(delta_out);

        out_layer.neurons[i].batch_pdb += delta_out;
        for (int w = 0; w < out_layer.neurons[i].weights.size(); ++w)
            out_layer.neurons[i].batch_pdw[w] += delta_out * out_layer.input[w];
    }

    // hidden layer
    for (int j = 0; j < hid_layer.neuron_num; ++j) {
        float delta_hid_tmp = 0;
        for (int k = 0; k < out_layer.neuron_num; ++k)
            delta_hid_tmp += deltas_out[k] * out_layer.neurons[k].weights[j];

        float delta_hid = delta_hid_tmp * hid_layer.neurons[j].GetDSigMoid();

        hid_layer.neurons[j].batch_pdb += delta_hid;
        for (int w = 0; w < hid_layer.neurons[j].weights.size(); ++w)
            hid_layer.neurons[j].batch_pdw[w] += delta_hid * hid_layer.input[w];
    }

    return;
}

void Net::UpdateMiniBatch(vector<VF>& batch_x, vector<VF>& batch_y)
{
    int batch_size = batch_x.size();
    hid_layer.ResetBatchPd(num_in);
    out_layer.ResetBatchPd(num_hid);
    for (int i = 0; i < batch_size; ++i) {
        Forward(batch_x[i]);
        BackPropagation(batch_y[i]);
    }

    // update bias and weights:
    for (int i = 0; i < out_layer.neuron_num; ++i) {
        out_layer.neurons[i].bias -= lr * out_layer.neurons[i].batch_pdb;
        for (int w = 0; w < out_layer.neurons[i].weights.size(); ++w)
            out_layer.neurons[i].weights[w] -= lr * out_layer.neurons[i].batch_pdw[w];
    }

    for (int i = 0; i < hid_layer.neuron_num; ++i) {
        hid_layer.neurons[i].bias -= lr * hid_layer.neurons[i].batch_pdb;
        for (int w = 0; w < hid_layer.neurons[i].weights.size(); ++w)
            hid_layer.neurons[i].weights[w] -= lr * hid_layer.neurons[i].batch_pdw[w];
    }
}

float Net::CalcErrOnce(VF& input, VF& label) {
    float err = 0, tmp = 0;
    Forward(input);
    for (int i = 0; i < label.size(); ++i) {
        tmp = out_layer.neurons[i].output - label[i];
        err += 0.5 * tmp * tmp;
    }

    return err;
}

float Net::CalcErrTotal(vector<VF> &input_x, vector<VF> &input_y) {
    float err_total = 0;
    int sample_num = input_x.size();
    for (int i = 0; i < sample_num; ++i)
        err_total += CalcErrOnce(input_x[i], input_y[i]);

    return err_total;
}

float Net::TrainNet(vector<VF> &inputs, vector<VF> &labels,
    int batch_size, float loss_critera/* = 0.002*/, int max_epoch/* = 10000*/) {
    int sample_num = inputs.size();
    batch_size = batch_size > sample_num ? sample_num : batch_size;

    int batch_per_epoch = sample_num / batch_size;
    int batch_left = sample_num % batch_size;
    float err_avg = 0;

    DataShuffler dtmng(sample_num);
    vector<VF> batchx, batchy;
    batchx.resize(batch_size);
    batchy.resize(batch_size);
    int bptr = 0;

    int epoch = 0, batch = 0, bleft = 0, rnd = 0;
    for (; epoch < max_epoch; ++epoch) {
        //shuffle data every epoch
        dtmng.ShuffleData(inputs, labels);

        if (epoch > 1000)
            lr = 0.03;
        else if (epoch > 2000)
            lr = .01;

        for (batch = 0; batch < batch_per_epoch; ++batch) {
            // get_batches and train
            bptr = batch * batch_size;
            batchx.assign(inputs.begin() + bptr, inputs.begin() + bptr + batch_size);
            batchy.assign(labels.begin() + bptr, labels.begin() + bptr + batch_size);
            UpdateMiniBatch(batchx, batchy);

            //err_avg = CalcErrTotal(inputs, labels) / sample_num;
            //cout << "batch " << batch << ", average loss: " << err_avg << endl;
        }

        if (batch_left > 0) {
            batchx.assign(inputs.begin() + bptr + batch_size, inputs.end());
            batchy.assign(labels.begin() + bptr + batch_size, labels.end());
            batchx.resize(batch_size);
            batchy.resize(batch_size);
            int fill_num = batch_size - batch_left;
            for (bleft = 0; bleft < fill_num; ++bleft) {
                rnd = rand() % sample_num;
                batchx[batch_left + bleft] = inputs[rnd];
                batchy[batch_left + bleft] = labels[rnd];
            }
        }

        err_avg = Net::CalcErrTotal(inputs, labels) / sample_num;
        if (0 == epoch % 100)
            cout << "epoch " << epoch << ", average loss: " << err_avg << endl;
        if (err_avg < loss_critera) {
            cout << "finish with loss criterion " << loss_critera << " at epoch " << epoch << endl;
            break;
        }

    }

    if (epoch == max_epoch)
        cout << "finish at max epoch " << epoch << endl;

    return err_avg;
}
