#include <iostream>
#include <Eigen/Dense>
#include "nn.cpp"

int main()
{
    // CREAR LA ANN 
    NN red_neuronal;
    /*
    NN       ESTRUCTURA DE OBJETO:
    │
    └─<layers> 
        │
        ├──[0] unique_ptr → layer A
        ├──[1] unique_ptr → layer B
        └──[2] unique_ptr → layer C
    */
    

    // AÑADIR CAPAS
    // 1. new Type() crea un objeto type en el heap y devuelve el puntero [ las capas heredan el tipo de la clase Layer() ]
    // 2. nn.add() recibe el puntero y lo guarda en el vector de punteros inteligentes [ nn.layers ]
    
    red_neuronal.add(new Linear(2,3)); // Linear(neuronas de entrada, neuronas de salida)
    red_neuronal.add(new Relu());

    red_neuronal.add(new Linear(3,6));
    red_neuronal.add(new Tanh());

    red_neuronal.add(new Linear(6,8));
    red_neuronal.add(new Sigmoid());

    red_neuronal.add(new Linear(8,3));
    red_neuronal.add(new Tanh());

    red_neuronal.add(new Linear(3,1));
    red_neuronal.add(new Sigmoid());


    // INPUT DATA 
    std::vector< Eigen::VectorXd > x(4);
    x[0] = Eigen::Vector2d(0,0);
    x[1] = Eigen::Vector2d(0,1);
    x[2] = Eigen::Vector2d(1,0);
    x[3] = Eigen::Vector2d(1,1);

    std::vector< Eigen::VectorXd > y(4);
    y[0] = Eigen::VectorXd::Constant(1, 0.0);
    y[1] = Eigen::VectorXd::Constant(1, 1.0);
    y[2] = Eigen::VectorXd::Constant(1, 1.0);
    y[3] = Eigen::VectorXd::Constant(1, 0.0);

    // NETWORK TRAIN
    red_neuronal.Train2D(x,y,30000,.05);

    // TESTING 
    Eigen::Vector2d input(0,0);
    Eigen::VectorXd output_prob = red_neuronal.predict(input);
    double output_val = 0;

    if (output_prob[0] > 0.5)
    {
        output_val = 1;
    }
    else
    {
        output_val = 0;
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output_val << std::endl;
    std::cout << "Expected Output: " << 0 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = Eigen::Vector2d(0, 1);
    output_prob = red_neuronal.predict(input);
    if (output_prob[0] > 0.5)
    {
        output_val = 1;
    }
    else
    {
        output_val = 0;
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output_val << std::endl;
    std::cout << "Expected Output: " << 1 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = Eigen::Vector2d(1, 0);
    output_prob = red_neuronal.predict(input);
    if (output_prob[0] > 0.5)
    {
        output_val = 1;
    }
    else
    {
        output_val = 0;
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output_val << std::endl;
    std::cout << "Expected Output: " << 1 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = Eigen::Vector2d(1, 1);
    output_prob = red_neuronal.predict(input);
    if (output_prob[0] > 0.5)
    {
        output_val = 1;
    }
    else
    {
        output_val = 0;
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output_val << std::endl;
    std::cout << "Expected Output: " << 0 << std::endl;
    std::cout << "----------------------" << std::endl;

    return 0;

}