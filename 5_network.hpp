#pragma once 

#include <Eigen/Dense>
#include <memory>
#include <iostream>

#include "4_layers.hpp"
#include "3_loss.hpp"


class NN 
{
public: 
    Eigen::MatrixXd IC_trueLabel;
    Eigen::MatrixXd BC_trueLabel;
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

    
    void Export(std::string filename)
    {
        Eigen::VectorXd linear_layers {{0,2,4,6,8}};
        std::ofstream outputfile(filename);

        for(int i = 0; i < linear_layers.size(); i++)
        {
            outputfile << "> Capa " <<linear_layers(i)<< ":" << "\n";
            Eigen::MatrixXd weightMatrix = layers[linear_layers(i)]-> W;
            Eigen::MatrixXd biasVector = layers[linear_layers(i)]-> b;

            outputfile << "W: \n";
            writeMatrix(outputfile,weightMatrix) ;
            outputfile << "\n";

            outputfile << "b: \n";
            writeVector(outputfile, biasVector);
            outputfile << "\n";

            if(linear_layers(i) != 8)
            {
                outputfile << "> Capa " <<linear_layers(i)+1<< ": tanh" << "\n";
                outputfile << "\n";
                
            }
        }
        outputfile.close();
    }


    void add(Layer* layer)
    {
        layers.emplace_back(layer);
    }

    Eigen::VectorXd predict(Eigen::VectorXd input)
    {
        return forward_propagation(input);
    }

    Eigen::VectorXd forward_propagation(const Eigen::VectorXd input)
    {
        int d = input.size();

        // 1. Forward clásico de activaciones
        Eigen::VectorXd output = input;
        for (const auto &layer : layers)
        {
            output = layer->forward(output);
        }

        // 2. Propagación de derivadas (Jacobianos y Hessianos respecto a x)
        Eigen::MatrixXd current_J = Eigen::MatrixXd::Identity(d, d);
        std::vector<Eigen::MatrixXd> current_H(d, Eigen::MatrixXd::Zero(d, d));

        for (const auto &layer : layers)
        {
            layer->forward_derivatives(current_J, current_H);
            current_J = layer->layerJacobian;
            current_H = layer->layerHessian;
        }

        NetworkJacobian_f = current_J;
        NetworkHessianTensor = current_H;

        return output; 
    }

    void backward_propagation(const LossGradients &grads)
    {
        LossGradients current_grad = grads;
        for (auto &H : current_grad.d_H)
        {
            if (H.rows() > 0 && H.cols() > 0)
            {
                H = 0.5 * (H + H.transpose());
            }
        }

        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            current_grad = (*it)->backward(current_grad);
        }
    }

    void zero_grad()
    {
        for (auto &layer : layers)
        {
            layer->zero_grad();
        }
    }

    void step(double learning_rate)
    {
        for (auto &layer : layers)
        {
            layer->step(learning_rate);
        }
    }

    void backward_propagation(Layer::backwardOutput & derivatives, double learning_rate)
    {
        Layer::backwardOutput grad  = derivatives;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad.error, grad.sensitivity, learning_rate);
        }
    }

    void Train(Eigen::MatrixXd IC_networkInput, Eigen::MatrixXd BC_networkInput, Eigen::MatrixXd EQ_networkInput,double w_ic,double w_bc,double w_eq, int epochs, double learning_rate, double errorChangeThreshold, double factor)
    {
        int d = IC_networkInput.cols();
        double prevError = 0;

        // Ejecutar un forward inicial para conocer la dimensión de salida y preparar capas
        Eigen::VectorXd sample_out = forward_propagation(IC_networkInput.row(0));
        int n_out = sample_out.size();

        int n_ic = IC_networkInput.rows();
        int n_bc = BC_networkInput.rows();
        int n_eq = EQ_networkInput.rows();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            zero_grad();

            //Condiciones iniciales (IC) ----------------------------------------------------
            double IC_totalError = 0;
            for (int n = 0; n < n_ic; n++)
            {
                Eigen::VectorXd u_pred = forward_propagation(IC_networkInput.row(n));
                Eigen::VectorXd u_real(n_out);
                if (IC_trueLabel.cols() == n_out) {
                    u_real = IC_trueLabel.row(n);
                } else {
                    u_real[0] = IC_trueLabel(n);
                }

                IC_totalError += MSEloss(u_real, u_pred);

                // Gradiente de la pérdida IC respecto a la salida
                LossGradients ic_grads(n_out, d);
                ic_grads.d_out = w_ic * MSElossDerivative(u_real, u_pred) / n_ic;
                backward_propagation(ic_grads);
            }
            double IC_error = IC_totalError / n_ic;

            //Condiciones de frontera (BC) -------------------------------------------------
            double BC_totalError = 0;
            for (int n = 0; n < n_bc; n++)
            {
                Eigen::VectorXd u_pred = forward_propagation(BC_networkInput.row(n));
                Eigen::VectorXd u_real(n_out);
                if (BC_trueLabel.cols() == n_out) {
                    u_real = BC_trueLabel.row(n);
                } else {
                    u_real[0] = BC_trueLabel(n);
                }

                BC_totalError += MSEloss(u_real, u_pred);

                // Gradiente de la pérdida BC respecto a la salida
                LossGradients bc_grads(n_out, d);
                bc_grads.d_out = w_bc * MSElossDerivative(u_real, u_pred) / n_bc;
                backward_propagation(bc_grads);
            }
            double BC_error = BC_totalError / n_bc;

            //Residuo físico (PDE: Ecuación de onda 2D) -----------------------------------
            double Residual = 0;
            for (int n = 0; n < n_eq; n++)
            {
                forward_propagation(EQ_networkInput.row(n));
                double u_xx = NetworkHessianTensor[0](0, 0);
                double u_yy = NetworkHessianTensor[0](1, 1);
                double u_tt = NetworkHessianTensor[0](2, 2);

                double residual_n = (u_xx + u_yy) - u_tt;
                Residual += pow(residual_n, 2);

                // =========================================================================
                // DEFINICIÓN DE LA DERIVADA DEL RESIDUO Y DE LA PÉRDIDA FÍSICA:
                // L_res = residual_n^2  ==>  dL/d(residual) = (2 * residual_n) / n_eq
                // =========================================================================
                double dL_dres = w_eq *(2.0 * residual_n) / n_eq;

                LossGradients eq_grads(n_out, d);
                // Similar a la definición del residuo: residual_n = (u_xx + u_yy) - u_tt
                eq_grads.d_H[0](0, 0) = dL_dres * (1.0);  // término u_xx
                eq_grads.d_H[0](1, 1) = dL_dres * (1.0);  // término u_yy
                eq_grads.d_H[0](2, 2) = dL_dres * (-1.0); // término -u_tt

                backward_propagation(eq_grads);
            }
            Residual = Residual / n_eq;
            double totalError = IC_error + BC_error + Residual;
            double errorChange = std::abs((totalError-prevError)/prevError)*100;
            prevError = totalError;

            // Optimización: actualización de pesos y sesgos por Descenso de Gradiente
            step(learning_rate);

            if (epoch % 1 == 0 || epoch == epochs - 1)
            {
                std::cout << "Epoch: " << epoch 
                          << "     | Total error: " << totalError  
                          << "     | IC error: " << IC_error 
                          << "     | BC error: " << BC_error 
                          << "     | Residual: " << Residual 
                          << "     | Error Change: " << errorChange << "%"
                          << "     | Learning Rate: " << learning_rate << std::endl;;
            }


        }
    }
};
