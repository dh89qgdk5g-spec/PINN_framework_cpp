#include <iostream>
#include "5_network.hpp"
#include "import_utils.cpp"



int main()
{
    //import de datos:
    Eigen::MatrixXd data = loadCSV("numericDataset.csv");

    Eigen::MatrixXd datos = Eigen::MatrixXd::Ones(12,3);
    Eigen::VectorXd input(3);
    input << 1,2,3;

    NN Ns;
    Ns.add(new Linear(3,24));
    Ns.add(new Sigmoid ());

    Ns.add(new Linear(24,12));
    Ns.add(new Tanh());

    Ns.add(new Linear(12,3));



    //Ns.predict(input);
    //Ns.TrainVector(datos,datos, 3000, 0.0001);
    //std::cout << Ns.predict(input).transpose() << std::endl;
    std::cout << data << std::endl;

    return 0;
}
