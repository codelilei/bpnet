/***************************************************************
## Implementation of 3-layer BP net
   to fit sin function as an example
## Author: Andrew Lee
## Date  : July 10th, 2017
## Email : code.lilei@gmail.com
***************************************************************/

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
using namespace cv;
#endif

#include <time.h>
#include <math.h>

#include <iostream>
#include <string>
#include <vector>

#include "data.hpp"
#include "net.hpp"


using std::vector;
using std::string;


void main()
{
    srand((unsigned)time(NULL));

    float lr = .05;
    float loss_thresh = .0015;
    int batch_size = 8;

#if 0
    VF input_x;
    //sample test
    input_x.push_back(0.05);
    input_x.push_back(0.1);
    vector<VF> inputs_x;
    inputs_x.push_back(input_x);

    VF input_y;
    input_y.push_back(0.01);
    input_y.push_back(0.99);
    vector<VF> inputs_y;
    inputs_y.push_back(input_y);

    Net net(0.5, 2, 2, 2);
    net.TrainNet(inputs_x, inputs_y, batch_size, loss_thresh);

#else
    vector<VF> inputs_x, inputs_y;
    DataReader train_data("../train.txt");
    train_data.ReadData(inputs_x, inputs_y);
    //train_data.ReadData(input_x, input_y);
    //train_data.NormData(input_x, input_y);

    Net net(lr, 1, 10, 1);
    net.TrainNet(inputs_x, inputs_y, batch_size, loss_thresh);

#ifdef USE_OPENCV
    //show fit effect in graph
    vector<VF> test_x, test_y;
    DataReader test_data("../test.txt");
    test_data.ReadData(test_x, test_y);

    float zoom = 50, y_out = 0, err = 0;
    int h = 8 * zoom, w = 16 * zoom;

    Mat img(h, w, CV_8UC3, Scalar::all(255));
    VF vout;
    vector<VF>::iterator itx = test_x.begin(), ity = test_y.begin();
    for (; itx != test_x.end() && ity != test_y.end(); ++itx, ++ity) {
        vout = net.Forward(*itx);
        y_out = vout[0];
        err += net.CalcErrOnce(*itx, *ity);
        circle(img, Point(zoom * ((*itx)[0] + 8), h - zoom * ((*ity)[0] + 4)), 1, Scalar(0, 255, 0));
        circle(img, Point(zoom * ((*itx)[0] + 8), h - zoom * (y_out + 4)), 1, Scalar(0, 0, 255));
    }

    line(img, Point(0.8 * w, 0.1 * h), Point(0.85 * w, 0.1 * h), Scalar(0, 255, 0), 2);
    line(img, Point(0.8 * w, 0.15 * h), Point(0.85 * w, 0.15 * h), Scalar(0, 0, 255), 2);
    putText(img, "Actual", Point(0.88 * w, 0.1 * h), CV_FONT_HERSHEY_COMPLEX, .5, Scalar(0, 255, 0), 1);
    putText(img, "Fit", Point(0.88 * w, 0.15 * h), CV_FONT_HERSHEY_COMPLEX, .5, Scalar(0, 0, 255), 1);
    putText(img, "Loss: " + std::to_string(err / test_x.size()), Point(0.4 * w, 0.12 * h), 2, .5, Scalar(255, 0, 0), 1);
    imshow("BPFit", img);
    imwrite("result.jpg", img);
    waitKey(0);
#endif

#endif

    return;
}
