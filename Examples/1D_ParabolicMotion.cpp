#include <iostream> 
#include <cmath>
#include "1_utils.hpp"

#include "4_layers.hpp"
#include "5_network.hpp"


int main(){
    // test data generation:

    Eigen::VectorXd t = vInit(0,6,100);
    Eigen::VectorXd h = t;

    for (int i = 0; i < h.size(); i++)
    {
        h[i] = 3*h[i] - .5 * pow(h[i], 2);
    }
    //vprint(t);
    //vprint(h);


    NN ann;

    ann.add(new Linear(1,24));
    ann.add(new Tanh());

    ann.add(new Linear(24,10));
    ann.add(new Tanh());

    ann.add(new Linear(10,1));
    
    
    ann.Train1D(t,h,100000,0.0002);
    
    Eigen::VectorXd input(1);
    input[0] = 3.0;
    double salida = ann.predict(input)[0];

    
    //std::cout << std::endl;
    //std::cout << "Prediction: " << std::endl;
    //std::cout << "{  " << salida << "  }" << std::endl;
    //std::cout << 3*3 -.5* pow(3,2) << std::endl;

    std::cout << "Network Jacobian: " << std::endl;
    mprint(ann.NetworkJacobian_b);

    ann.Train1D(t, h, 10000, 0.0002);
}

