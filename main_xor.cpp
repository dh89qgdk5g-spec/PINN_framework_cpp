#include <iostream>
#include <vector>
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
    std::vector< std::vector<double> > x = {{0,0},{0,1},{1,0},{1,1}};
    std::vector< std::vector<double> > y = {{0},{1},{1},{0}};

    // NETWORK TRAIN
    red_neuronal.fit(x,y,30000,.05);

    // TESTING 
    std::vector<double> input = {0,0};
    std::vector<double> output_prob = red_neuronal.predict(input);
    std::vector<double> output = {0};

    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 0 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = {0, 1};
    output_prob = red_neuronal.predict(input);
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 1 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = {1, 0};
    output_prob = red_neuronal.predict(input);
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 1 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = {1, 1};
    output_prob = red_neuronal.predict(input);
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 0 << std::endl;
    std::cout << "----------------------" << std::endl;

    return 0;

}