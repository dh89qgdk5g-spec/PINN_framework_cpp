#pragma once
#include <Eigen/Dense>
#include <iostream>

#include "utils_1.cpp"
#include "activation_1.cpp"


class Layer 
{
public: 
    Eigen::VectorXd input;
    Eigen::VectorXd output;

    Eigen::MatrixXd jacobian_1;

    virtual Eigen::VectorXd forward(const Eigen::VectorXd input_data) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) = 0; 
    //backward(∂L/∂ŷ , \nu)

};

class Sigmoid : public Layer 
{
    public:
    Eigen::VectorXd derivative_1;

    Eigen::VectorXd forward (const Eigen::VectorXd input_data) override
    {
        //ANN
        input = input_data;
        output = vectSigmoid(input);

        //PINN 
        derivative_1 = vectSigmoidDerivative(input);
        jacobian_1 = buildJacobian(derivative_1);
        
        return output;
    }
    
    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
};

class Relu : public Layer 
{
    public:
    Eigen::VectorXd derivative_1;
    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    
    {
        //ANN
        input  = input_data;
        output = vectRelu(input);

        //PINN
        derivative_1 = vectReluDerivative(input);
        jacobian_1 = buildJacobian(derivative_1);

        return output;
    }

    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
}; 

class LeakyRelu : public Layer
{
    public: 
    double alpha = 0.01;
    Eigen::VectorXd derivative_1;

    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        //ANN
        input = input_data;
        output = vectLeakyRelu(input,alpha);

        //PINN
        derivative_1 = vectLeakyReluDerivative(input,alpha);
        jacobian_1 = buildJacobian(derivative_1);

        
        return output; 
    }

    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
};

class Tanh : public Layer 
{
    public:
    Eigen::VectorXd derivative_1;
    
    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        //ANN
        input = input_data;
        output = vectTanh(input);

        //PINN
        derivative_1 = vectTanhDerivative(input);
        jacobian_1 = buildJacobian(derivative_1);
        return output;
    }

    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
};






class Linear : public Layer 
{
    public:
    int input_neuron;
    int output_neuron;

    // network components
    Eigen::MatrixXd weights; // Dimensión: (num_out, num_in)
    Eigen::VectorXd bias;    // Dimensión: (num_out)

    // Physics components
    
    //previous layer {a} components
    Eigen::VectorXd a;
    Eigen::MatrixXd ax;

    Eigen::MatrixXd aw;
    Eigen::matrixXd ab;

    Eigen::MatrixXd axw;
    Eigen::MatrixXd axb;

    //Local layer components
    Eigen::VectorXd v;
    Eigen::MatrixXd vx;
    
    Eigen::MatrixXd vw;
    Eigen::MatrixXd vb; 
    Eigen::MatrixXd vxw;
    Eigen::MatrixXd vxb;
    
    Eigen::MatrixXd forward_vw;
    Eigen::MatrixXd forward_vb;
    Eigen::MatrixXd forward_vxw;
    Eigen::MatrixXd forward_vxb;

    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitializer(num_out);
    }

    // Neural network
    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        input = input_data;
        output = weights * input + bias;

        /*
        No es realmente el jacobiano de Linear, pero se hace una copia de weights con este
        nombre para extraer el valor en NN_pinn.cpp de una manera mas sencila y poder hacer 
        la secuencia de Multiplicaciones du/dx. 
        */
        
        jacobian_1 = weights;
        return output;
    }
   
    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        // error = ∂e/∂z
        // learning_rate = η

        //gradiente respecto al input ∂L/∂x = w'@error
        Eigen::VectorXd input_error = weights.transpose() * error;

        //Gradiente respecto a los pesos ∂L/∂w = {e}@{x}'
        Eigen::MatrixXd weight_error = error * input.transpose();

        //gradiente respecto a bias  ∂L/∂b = error
        Eigen::VectorXd bias_error = error;

        //-------------------------------------------------------

        //actualización de bias 
        bias = bias - learning_rate * bias_error;

        //actualización de pesos 
        weights = weights - learning_rate * weight_error;

        return input_error;
    } 

    // Physics 
    void compute_gd_terms () 
    {
        v  = output;

        vx = weights * ax; // [w] @ {∂a/∂x}

        vw = Eigen::MatrixXd::Zero(output_neuron,input_neuron*output_neuron);// [ [∂z1/∂W] [∂z2/∂W] [∂z3/∂W] ]  
        for(int i = 0; i < output_neuron; i++) // ∂z_k/∂W_ij -> k==i ? a_j:0 
        {
            for (int j = 0; j < input_neuron; j++)
            {
                vw(i,j+(i*input_neuron)) = a(j);
            }
        }          
    }
};

