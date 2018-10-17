#ifndef BP_DATA_HPP_
#define BP_DATA_HPP_

#include <vector>

typedef std::vector<float> VF;


class DataReader {
 public:
    DataReader(std::string file) : file_name(file) {}

    void ReadData(std::vector<VF>& in_x, std::vector<VF>& in_y);
    void ReadData(VF& in_x, VF& in_y);
    void NormData(VF& in_x, VF& in_y);

 private:
    std::string file_name;
};


class DataShuffler {
 public:
    DataShuffler() {}
    DataShuffler(int n) {
        vidx.resize(n);
        for (int i = 0; i < n; ++i)
            vidx[i] = i;
        tmp_x.resize(n);
        tmp_y.resize(n);
    }

    void ShuffleData(std::vector<VF>& vsamples_x, std::vector<VF>& vsamples_y);

 private:
    std::vector<int> vidx;
    std::vector<VF> tmp_x;
    std::vector<VF> tmp_y;
};

#endif  // BP_DATA_HPP_
