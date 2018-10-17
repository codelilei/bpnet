#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "data.hpp"

using namespace std;


void DataReader::ReadData(vector<VF>& in_x, vector<VF>& in_y) {
    float fx = .0, fy = .0;

    ifstream ifile(file_name);
    if (!ifile) {
        cout << "file open error, u may check the file path" << endl;
        return;
    }

    while (ifile >> fx >> fy) {
        in_x.push_back(VF{ fx });
        in_y.push_back(VF{ fy });
    }
}

void DataReader::ReadData(VF& in_x, VF& in_y) {
    string line;
    float fx = .0, fy = .0;

    ifstream ifile(file_name);
    if (!ifile) {
        cout << "file open error, u may check the file path" << endl;
        return;
    }

    //while (getline(ifile, line))
    while (ifile >> fx >> fy) {
        //stringstream tmp(line);
        //tmp >> fx >> fy;

        //xy_pair.push_back(make_pair(fx, fy));
        in_x.push_back(fx);
        in_y.push_back(fy);
    }
}

void DataReader::NormData(VF& in_x, VF& in_y) {
    float min_x = *min_element(in_x.begin(), in_x.end());
    float max_x = *max_element(in_x.begin(), in_x.end());
    float min_y = *min_element(in_y.begin(), in_y.end());
    float max_y = *max_element(in_y.begin(), in_y.end());

    //normalize to [-1, 1]
    for (int i = 0; i < in_x.size(); ++i) {
        in_x[i] = (in_x[i] - min_x) / (max_x - min_x) * 2 - 1;
        in_y[i] = (in_y[i] - min_y) / (max_y - min_y) * 2 - 1;
    }
}


void DataShuffler::ShuffleData(vector<VF>& vsamples_x, vector<VF>& vsamples_y) {
    int n = vsamples_x.size();
    random_shuffle(vidx.begin(), vidx.end());
    //transform(vidx.begin(), vidx.end(), tmp_x.begin(), [&](int idx) {return vsamples_x[vidx[idx]]; });
    //transform(vidx.begin(), vidx.end(), tmp_y.begin(), [&](int idx) {return vsamples_y[vidx[idx]]; });
    for (int j = 0; j < n; ++j) {
        tmp_x[j] = vsamples_x[vidx[j]];
        tmp_y[j] = vsamples_y[vidx[j]];
    }
    vsamples_x.assign(tmp_x.begin(), tmp_x.end());
    vsamples_y.assign(tmp_y.begin(), tmp_y.end());
}
