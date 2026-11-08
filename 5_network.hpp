#pragma once 

#include <Eigen/Dense>
#include <memory>
#include <iostream>

#include "4_layers.hpp"
#include "3_loss.hpp"


class NN 
{
    public: 
    Eigen::VectorXd IC_trueLabel;
    Eigen::VectorXd BC_trueLabel;
    int EQ_points;


    Eigen::MatrixXd NetworkJacobian_f;
    std::vector<Eigen::MatrixXd> NetworkHessianTensor;




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
        //forward -------------------------------------------------------
        Eigen::VectorXd output = input;
        for (const auto &layer : layers) // en cada iteración &layer se convierte en una referencia a un elemento del vector {"layers"} [std::unique_ptr<layer>]
        {
            output = layer->forward(output); // ejecuta el método forward() del objeto al que apunta layer (layer que es una referencia a un elemento de la lista de punteros inteligentes)
        }


        //Jacobianos de capa y de red-----------------------------------------
        NetworkJacobian_f = layers.front() -> localJacobian;
        layers.front()-> layerJacobian = NetworkJacobian_f;
        Eigen::MatrixXd dynamicLayerJacobian = NetworkJacobian_f;

        for (int i = 1; i < layers.size(); i++)
        {
        NetworkJacobian_f = layers[i]->localJacobian * dynamicLayerJacobian;

        layers[i]->layerJacobian = NetworkJacobian_f;
        dynamicLayerJacobian = NetworkJacobian_f;
        }

        //Tensor de hessianos------------------------------------------------
        std::vector<Eigen::MatrixXd> networkHessianTensor(input.size());
        for (int i = 0; i < input.size(); i++)
        {
            networkHessianTensor[i] = Eigen::MatrixXd::Zero(input.size(),input.size());
        }
        
        for (int i = 0; i < layers.size(); i++)
        {
            networkHessianTensor = layers[i] -> calculateHessianTensor(networkHessianTensor);
        }
        NetworkHessianTensor = networkHessianTensor;

        return output; 
    }


    void backward_propagation(Layer::backwardOutput & derivatives, double learning_rate)
    {
        Layer::backwardOutput grad  = derivatives;

        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it) -> backward(grad.error, grad.sensitivity, learning_rate);
        }
    }

    void Train(Eigen::MatrixXd IC_networkInput, Eigen::MatrixXd BC_networkInput, Eigen::MatrixXd EQ_networkInput, int epochs, double learning_rate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            //Condiciones iniciales (IC) ----------------------------------------------------
            double IC_totalError = 0;
            for (int n = 0; n < IC_networkInput.rows(); n++)
            {
                Eigen::VectorXd u_pred = forward_propagation(IC_networkInput.row(n));
                Eigen::VectorXd u_real = IC_trueLabel.row(n);

                IC_totalError += MSEloss(u_real, u_pred);
            }
            double IC_error = IC_totalError / IC_networkInput.rows();

            //Condiciones de frontera (BC) -------------------------------------------------
            double BC_totalError = 0;
            for (int n = 0; n < BC_networkInput.rows(); n++)
            {
                Eigen::VectorXd u_pred = forward_propagation(BC_networkInput.row(n));
                Eigen::VectorXd u_real = BC_trueLabel.row(n);

                BC_totalError += MSEloss(u_real, u_pred);
            }
            double BC_error = BC_totalError / BC_networkInput.rows();

            //Residuo físico (PDE) --------------------------------------------------------
            double Residual = 0;
            for (int n = 0; n < EQ_networkInput.rows(); n++)
            {
                forward_propagation(EQ_networkInput.row(n));
                double u_xx = NetworkHessianTensor[0](0, 0);
                double u_yy = NetworkHessianTensor[0](1, 1);
                double u_tt = NetworkHessianTensor[0](2, 2);

                double residual_n = (u_xx + u_yy) - u_tt;
                Residual += pow(residual_n, 2);
            }
            Residual = Residual / EQ_networkInput.rows();

            std::cout << "Epoch: "<< epoch <<"     | Total error: " << IC_error + BC_error + Residual  <<"     | IC error: " << IC_error<<"     | BC error: "<< BC_error<<"     | Residual: " << Residual << std::endl;
        }
    }
};