#pragma once 

#include <vector>
#include <memory>
#include <iostream>
#include "layer_1.cpp"
#include "loss_1.cpp"

class NN 
{
    public: 
    std::vector< std::unique_ptr<Layer> > layers; // vector de punteros inteligentes que apuntan a objetos de tipo "Layer"
    /*
    NN       ESTRUCTURA DE OBJETO:
    │
    └─<layers> 
        │
        ├──[0] unique_ptr → layer A
        ├──[1] unique_ptr → layer B
        └──[2] unique_ptr → layer C
    */
    



    void add(Layer* layer)
    {
        layers.emplace_back(layer); // construye el elemento dentro del vector {"layers"} pasando 'layer' como argumento al contructor de unique_ptr<layer>
    }
    
    std::vector<double> predict(std::vector<double> input)
    {
        return forward_propagation(input);
    }

    std::vector<double> forward_propagation(const std::vector<double> input)
    {
        std::vector<double> output = input;
        for (const auto &layer : layers) // en cada iteración &layer se convierte en una referencia a un elemento del vector {"layers"} [std::unique_ptr<layer>]
        {
            output = layer->forward(output); // ejecuta el método forward() del objeto al que apunta layer (layer que es una referencia a un elemento de la lista de punteros inteligentes)
        }
        return output; 
    }

    void backward_propagation(const std::vector<double> &error, double learning_rate)
    {
        std::vector<double> grad = error; // permite tener una copia de error en grad y conservar el acceso al origial mediante error
        for (auto it = layers.rbegin(); it != layers.rend(); ++it ) // rbegin() me dice que el iterador comienza al final y aumenta en sentido contrario 
        {
            // [it] funciona como un puntero que apunta a un elemento del vector y [*it] accede al contenido de esa posición 
            grad = (*it) -> backward(grad, learning_rate);
        }
    }

    void Train2D(const std::vector< std::vector<double> > &X, const std::vector< std::vector<double>> &y, int epochs, double learning_rate ) 
    {
        for (int epoch = 0; epoch < epochs ; ++epoch) 
        {
            double total_loss = 0. ;

            
            for (size_t i = 0; i < X.size(); ++i) // por cada epoch  recorre los [i] elementos de x, y con el elemento [i] hace fprop y bprop
            {
                //forward pass 
                std::vector<double> output = forward_propagation(X[i]);

                // Calcular pérdida
                double loss = BCEloss(y[i], output);
                total_loss += loss;

                std::vector<double> loss_derivative = BCElossDerivative(y[i], output);
                // backward pass 
                backward_propagation(loss_derivative, learning_rate);
            }
        


            // print loss 
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - loss: " << total_loss / X.size() << std::endl; 
        }
    }

    void Train1D (const std::vector<double> &X, const std::vector<double> &Y, int epochs, double learning_rate)
    {
    
        for (int epoch = 0; epoch < epochs ; ++epoch)
        {
            double total_loss = 0.;
            
            for (int i  = 0; i <X.size(); ++i)
            {
                //f.prop
                std::vector<double> output = forward_propagation(std::vector<double>{X[i]});
                //std::cout << Y[i] - output[0] << std::endl;
                // calcular pérdida
                double loss = MSEloss(std::vector<double>{Y[i]},output);
                total_loss += loss;
                std::vector<double> loss_derivative = MSElossDerivative(std::vector<double>{Y[i]},output); 
    
                //b.prop 
                backward_propagation(loss_derivative, learning_rate);

            }

            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - loss: " << total_loss/X.size()  << std::endl; 
        }
    }

    std::vector<double> Derivative_BackPropagation()
    {
        /*
        proceso a implementar:

        ultimas capas:
            z = Wx + b
            u = s(z) 
        
        derivada de la salida respecto a la entrada: 
            du/dx = du/dz * dz/dx 
            du/dx = s'(z) * w
        */
        

        std::vector<double> field_variables = layers.back() -> input; //extrae el valor con el que se evalua la ultima función de activación(z) 
        
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            field_variables = (*it) -> derivative_1_backpropagation(field_variables); //se evalua con las funciones derivadas en orden inverso
        }
        return field_variables; 
    }


    
};