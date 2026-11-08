#pragma once
#include <Eigen/Dense>
#include <vector>

#include "1_utils.hpp"
#include "2_activations.hpp"

class Layer
{
public:
    Eigen::VectorXd input;
    Eigen::VectorXd output;

    // 1st derivatives
    Eigen::MatrixXd localJacobian;
    Eigen::MatrixXd layerJacobian;
    
    // 2nd derivatives
    std::vector <Eigen::MatrixXd> hessian_2; 
    
    Eigen::MatrixXd jacobian_1;

    struct backwardOutput
    {
        //acceso por fura de una clase layer, mediente Layer::backwardOutput
        Eigen::VectorXd error; // ∂L/∂x propagador de perdida de datos
        Eigen::VectorXd sensitivity; // ∂u/∂a propadagor de perdida física
    };
    backwardOutput propagations;

    virtual Eigen::VectorXd forward(const Eigen::VectorXd input_data) = 0;
    virtual backwardOutput backward(Eigen::VectorXd error,Eigen::VectorXd sensitivity, double learning_rate) = 0; 
    virtual std::vector <Eigen::MatrixXd> calculateHessianTensor(std::vector <Eigen::MatrixXd> prev_hessian) = 0;
    //backward(∂L/∂ŷ , \nu)

};

class Sigmoid : public Layer 
{
    public:
    Eigen::VectorXd derivative_1;


    std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_Hessian) override
    {
        std::vector<Eigen::MatrixXd> layerHessians;
        
        for (int i = 0; i < output.size(); i++)
        {
            Eigen::RowVectorXd Ji = layerJacobian.row(i);
            Eigen::MatrixXd Jt_J = Ji.transpose()*Ji;

            layerHessians.push_back(sigmoid2Derivative(input(i))*Jt_J + sigmoidDerivative(input(i))*prev_Hessian[i]);
        }
        return layerHessians;
    }



    Eigen::VectorXd forward (const Eigen::VectorXd input_data) override
    {
        //ANN
        input = input_data;
        output = vectSigmoid(input);

        //PINN 
        derivative_1 = vectSigmoidDerivative(input);
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = buildJacobian(derivative_1);
        
        return output;
    }
    
    /*
    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        //training  data
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        
        //phisic restrictions
        return grad_input;
    }
    */
    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        //training data
        propagations.error = derivative_1.cwiseProduct(error);

        //physics
        propagations.sensitivity = derivative_1.cwiseProduct(sensitivity);

        return propagations;
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
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = buildJacobian(derivative_1);

        return output;
    }

    /*
    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
    */
    
    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        //training data
        propagations.error = derivative_1.cwiseProduct(error);

        //physics
        propagations.sensitivity = derivative_1.cwiseProduct(sensitivity);

        return propagations;
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
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = buildJacobian(derivative_1);

        
        return output; 
    }

    /*
    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
    */

    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        //training data
        propagations.error = derivative_1.cwiseProduct(error);

        //physics
        propagations.sensitivity = derivative_1.cwiseProduct(sensitivity);

        return propagations;
    }
};

class Tanh : public Layer 
{
    public:
    Eigen::VectorXd derivative_1;

    std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_Hessian) override
    {
        std::vector<Eigen::MatrixXd> layerHessians;
        
        for (int i = 0; i < output.size(); i++)
        {
            Eigen::RowVectorXd Ji = layerJacobian.row(i);
            Eigen::MatrixXd Jt_J = Ji.transpose()*Ji;

            layerHessians.push_back(tanh2Derivative(input(i))*Jt_J + tanhDerivative(input(i))*prev_Hessian[i]);
        }
        return layerHessians;
    }
    
    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        //ANN
        input = input_data;
        output = vectTanh(input);

        //PINN
        derivative_1 = vectTanhDerivative(input);
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = buildJacobian(derivative_1);
        return output;
        
    }

    /*
    Eigen::VectorXd backward(Eigen::VectorXd error, double learning_rate) override
    {
        Eigen::VectorXd grad_input = derivative_1.cwiseProduct(error);
        return grad_input;
    }
    */
    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        //training data
        propagations.error = derivative_1.cwiseProduct(error);

        //physics
        propagations.sensitivity = derivative_1.cwiseProduct(sensitivity);

        return propagations;
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
   
    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitializer(num_out);
    }
    std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_Hessian) override
    {
       return tensorTransform(weights, prev_Hessian);
    }

    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        input = input_data;
        output = weights * input + bias;

        /*
        No es realmente el jacobiano de Linear, pero se hace una copia de weights con este
        nombre para extraer el valor en NN_pinn.cpp de una manera mas sencila y poder hacer 
        la secuencia de Multiplicaciones du/dx. 
        */
        
        localJacobian = weights;
        jacobian_1 = weights;
        return output;
    }


    
   
    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        //-------------------datos--------------------------------
        // error = ∂e/∂z
        //gradiente respecto al input ∂L/∂x = w'@error
        Eigen::VectorXd input_error = weights.transpose() * error;
            propagations.error = input_error;

        //Gradiente respecto a los pesos ∂L/∂w = {e}@{x}'
        Eigen::MatrixXd weight_error = error * input.transpose();

        //gradiente respecto a bias  ∂L/∂b = error
        Eigen::VectorXd bias_error = error;

        //--------------------fisica----------------------------
        //sensitivity = ∂a/∂u = w'@error
        Eigen::VectorXd input_sensitivity = weights.transpose() * sensitivity;
            propagations.sensitivity = input_sensitivity;

        //∂u/∂w = {sensitivity}@{x}'
        Eigen::MatrixXd u_w = sensitivity * input.transpose();

        //∂u/∂b = error
        Eigen::VectorXd u_b = sensitivity;

      //--------------------optimización-----------------------------
        //actualización de bias 
        bias = bias - learning_rate * bias_error;

        //actualización de pesos 
        weights = weights - learning_rate * weight_error;

        return propagations;
    } 
};