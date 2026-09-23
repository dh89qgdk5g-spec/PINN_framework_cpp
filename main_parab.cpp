#include <iostream> 
#include "utils_1.cpp"
#include "nn.cpp"



int main(){
    // test data generation:

    std::vector<double> t = vInit(0,6,100);
    std::vector<double> h = t;

    std::transform(h.begin(),h.end(),h.begin(),[](double h){return 3*h -.5* pow(h,2);});
    vprint(t);
    vprint(h);


    NN ann;

    ann.add(new Linear(1,8));
    ann.add(new Relu());

    ann.add(new Linear(8,5));
    ann.add(new Sigmoid());

    ann.add(new Linear(5,1));
    
    
    
    ann.Train1D(t,h,20000,.001);
    
    std::vector<double> salida;
    salida.push_back( ann.predict(std::vector<double>{3.})[0]);

    
    vprint(salida);
    std::cout << 3*3 -.5* pow(3,2) << std::endl;
    
}