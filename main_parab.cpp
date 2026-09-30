#include <iostream> 
#include <cmath>
#include "utils_1.cpp"
#include "NN_pinn.cpp"



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

    ann.add(new Linear(1,8));
    ann.add(new Relu());

    ann.add(new Linear(8,5));
    ann.add(new Sigmoid());

    ann.add(new Linear(5,1));
    
    
    ann.Train1D(t,h,2000,.001);
    
    Eigen::VectorXd input(1);
    input[0] = 3.0;
    double salida = ann.predict(input)[0];

    
    std::cout << std::endl;
    std::cout << "Prediction: " << std::endl;
    std::cout << "{  " << salida << "  }" << std::endl;
    std::cout << 3*3 -.5* pow(3,2) << std::endl;

    std::cout << "Network Jacobian: " << std::endl;
    mprint(ann.NetworkJacobian);

    
}