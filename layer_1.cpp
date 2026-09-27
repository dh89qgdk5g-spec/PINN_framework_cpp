#pragma once
#include <vector> 
#include <iostream>

#include "utils_1.cpp"
#include "activation_1.cpp"


class Layer 
{
public: 
    std::vector<double> input;
    std::vector<double> output;
    std::vector<std::vector<double>> jacobian_1;
    virtual std::vector<double> forward(const std::vector<double> input_data) = 0;
    virtual std::vector<double> backward(std::vector<double> error, double learning_rate) = 0; 
    //backward(∂L/∂ŷ , \nu)

};

class Sigmoid : public Layer 
{
    public:
    std::vector<double> derivative_1;

    std::vector<double> forward (const std::vector<double> input_data) override
    {
        //ANN
        input = input_data;
        output = vectSigmoid(input);

        //PINN 
        derivative_1 = vectSigmoidDerivative(input);
        jacobian_1 = buildJacobian(derivative_1);
        
        return output;
    }
    
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> grad_input;
        for (int i = 0; i < derivative_1.size(); i++)
        {
            grad_input.push_back(derivative_1[i]*error[i]);
        }
        return grad_input;
    }
};

class Relu : public Layer 
{
    public:
    std::vector<double> derivative_1;
    std::vector<double> forward(const std::vector<double> input_data) override
    
    {
        //ANN
        input  = input_data;
        output = vectRelu(input);

        //PINN
        derivative_1 = vectReluDerivative(input);
        jacobian_1 = buildJacobian(derivative_1);

        return output;
    }

    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> grad_input;
        for (int i = 0; i < derivative_1.size(); ++i)
        {
            grad_input.push_back(derivative_1[i] * error[i]);
        }   
        return grad_input;
    }
}; 

class LeakyRelu : public Layer
{
    public: 
    double alpha = 0.01;
    std::vector<double> derivative_1;

    std::vector<double> forward(const std::vector<double> input_data) override
    {
        //ANN
        input = input_data;
        output = vectLeakyRelu(input,alpha);

        //PINN
        derivative_1 = vectLeakyReluDerivative(input,alpha);
        jacobian_1 = buildJacobian(derivative_1);

        
        return output; 
    }

    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> grad_input;
        for (int i = 0; i<derivative_1.size(); i++)
        {
            grad_input.push_back(derivative_1[i] * error[i]);
        }
        return grad_input;
    }
};

class Tanh : public Layer 
{
    public:
    std::vector<double> derivative_1;
    
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        //ANN
        input = input_data;
        output = vectTanh(input);

        //PINN
        derivative_1 = vectTanhDerivative(input);
        jacobian_1 = buildJacobian(derivative_1);
        return output;
    }

    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> grad_input;
        for (int i = 0; i < derivative_1.size(); i++)
        {
            grad_input.push_back(derivative_1[i] * error[i]);
        }
        return grad_input;
    }
};

class Linear : public Layer 
{
    public:
    int input_neuron;
    int output_neuron;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitializer(num_out);
    }

    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output.clear();
        for (int i = 0; i < output_neuron; i++)
        {
            output.push_back(dotProduct(weights[i],input) + bias[i]);
        }

        /*
        No es realmente el jacobiano de Linear, pero se hace una copia de weights con este
        nombre para extraer el valor en NN_pinn.cpp de una manera mas sencila y poder hacer 
        la secuencia de Multiplicaciones du/dx. 
        */
        
        jacobian_1 = weights;
        return output;
    }
   
    //backpropagation 
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        // error = ∂e/∂z
        // learning_rate = η

        std::vector<double> input_error;
        std::vector<std::vector<double>> weight_error;
        std::vector<double> bias_error;
        std::vector<std::vector<double>> weight_transpose;

        weight_error.clear();
        bias_error.clear();
        input_error.clear();
        weight_transpose.clear();

        weight_transpose = transpose(weights);

        //gradiente respecto a bias  dL/db = error
        bias_error = error;

        //gradiente respecto al input dL/dx = w'@error
        for (int i = 0; i < weight_transpose.size(); i++)
        {
            input_error.push_back(dotProduct(weight_transpose[i],error));
        }

        //Gradiente respecto a los pesos e@x'
        for (int j = 0; j < error.size(); j++)
        {
            std::vector<double> row;
            for (int i = 0; i < input.size(); i++)
            {
                row.push_back(error[j] * input[i]);
            }
            weight_error.push_back(row);
        }

        //actualización de bias 
        std::vector<double> delta_bias = scalarVectorMultiplication(bias_error, learning_rate);
        bias = subtract(bias, delta_bias);

        //actualización de pesos 
        for (int i = 0; i < weight_error.size(); i++)
        {
            std::vector<double> delta_weight = scalarVectorMultiplication(weight_error[i],learning_rate);
            weights[i] = subtract(weights[i], delta_weight);
        }

        return input_error;
    } 
    

};


