#pragma once 

#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include "Eigen/src/Core/Matrix.h"
#include "layer_1.cpp"
#include "loss_1.cpp"

class NN 
{
    public: 
    Eigen::MatrixXd NetworkJacobian;
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

    Eigen::VectorXd predict(Eigen::VectorXd input)
    {
        return forward_propagation(input);
    }

    Eigen::VectorXd forward_propagation(const Eigen::VectorXd input)
    {
        Eigen::VectorXd output = input;

        for (const auto &layer : layers) // en cada iteración &layer se convierte en una referencia a un elemento del vector {"layers"} [std::unique_ptr<layer>]
        {
            output = layer->forward(output); // ejecuta el método forward() del objeto al que apunta layer (layer que es una referencia a un elemento de la lista de punteros inteligentes)
        }


        //Calculo del Jacobiano de Red
        NetworkJacobian = layers.back() -> jacobian_1;
        for(int i = layers.size()-2 ; i>=0; --i)
        {
            NetworkJacobian = NetworkJacobian * layers[i]->jacobian_1;
        }

        return output; 
    }

    /*
    void backward_propagation(const Eigen::VectorXd &error, double learning_rate)
    {
        Eigen::VectorXd grad = error; // permite tener una copia de error en grad y conservar el acceso al origial mediante error
        
        for (auto it = layers.rbegin(); it != layers.rend(); ++it ) // rbegin() me dice que el iterador comienza al final y aumenta en sentido contrario 
        {
            // [it] funciona como un puntero que apunta a un elemento del vector y [*it] accede al contenido de esa posición 
            grad = (*it) -> backward(grad, learning_rate);
        }
    }
    */


    void backward_propagation(Layer::backwardOutput & derivatives, double learning_rate)
    {
        Layer::backwardOutput grad  = derivatives;

        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it) -> backward(grad.error, grad.sensitivity, learning_rate);
        }
    }

    void Train2D(const std::vector< Eigen::VectorXd > &X, const std::vector< Eigen::VectorXd> &y, int epochs, double learning_rate ) 
    {
        for (int epoch = 0; epoch < epochs ; ++epoch) 
        {
            double total_loss = 0. ;

            
            for (size_t i = 0; i < X.size(); ++i) // por cada epoch  recorre los [i] elementos de x, y con el elemento [i] hace fprop y bprop
            {
                //forward pass 
                Eigen::VectorXd output = forward_propagation(X[i]);

                // Calcular pérdida
                double loss = BCEloss(y[i], output);
                total_loss += loss;

                Eigen::VectorXd loss_derivative = BCElossDerivative(y[i], output);
                // backward pass 
                backward_propagation(loss_derivative, learning_rate);
            }
        


            // print loss 
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - loss: " << total_loss / X.size() << std::endl; 
        }
    }

    void Train1D (const Eigen::VectorXd &X, const Eigen::VectorXd &Y, int epochs, double learning_rate)
    {
    
        for (int epoch = 0; epoch < epochs ; ++epoch)
        {
            double total_loss = 0.;
            
            for (int i  = 0; i <X.size(); ++i)
            {
                //f.prop
                Eigen::VectorXd input(1);
                input[0] = X[i];
                Eigen::VectorXd output = forward_propagation(input);
                //std::cout << Y[i] - output[0] << std::endl;
                               
                
                // calcular pérdida
                Eigen::VectorXd true_label(1);
                true_label[0] = Y[i];
                double loss = MSEloss(true_label, output);
                total_loss += loss;
               
                //b.prop 
                Eigen::VectorXd data_loss_derivative = MSElossDerivative(true_label, output);
                Eigen::VectorXd physics_loss_derivative = Eigen::VectorXd::Ones(data_loss_derivative.rows());

                Layer::backwardOutput losses;
                losses.error = data_loss_derivative;
                losses.sensitivity = physics_loss_derivative;
                
                backward_propagation(losses, learning_rate);

            }

            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - loss: " << total_loss/X.size()  << std::endl; 
        }
    }
    
};