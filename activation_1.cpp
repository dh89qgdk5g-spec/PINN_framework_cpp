#pragma once 


#include <cmath>
#include <vector>
#include <iostream>


double sigmoid(double x)
{
    /**
     * The sigmoid function maps any real-valued number to a value between 0 and 1.
     * It is often used in the output layer of a neural network when the task is a
     * binary classification problem.
     * @param x the input value
     * @return the output value of the sigmoid function
     */
    return 1 / (1+exp(-x));
}

double sigmoidDerivative(double x)
{
    /**
    * The derivative of the sigmoid function.
    * @param x the input value
    * @return the output value of the derivative of the sigmoid function
    */
    return exp(x) / pow((exp(x)+1),2);
}


std::vector<double> vectSigmoid(const std::vector<double> x)
{
    /**
     * A vectorized version of the sigmoid function.
     * @param x the input vector
     * @return a vector where each element is the sigmoid of the corresponding element in x
     */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoid(i));
    return result;
}

std::vector<double> vectSigmoidDerivative(const std::vector<double> x)
{
    /**
     * A vectorized version of the derivative of the sigmoid function.
     * @param x the input vector
     * @return a vector where each element is the derivative of the sigmoid of the corresponding element in x
     */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoidDerivative(i));
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double relu(double x)
{
    /**
    * The Rectified Linear Unit (ReLU) activation function.
    * @param x the input value
    * @return the output value of the ReLU function
    */ 

    return std::max(0.0,x);
}

double reluDerivative(double x)
{
    /**
    * The derivative of the Rectified Linear Unit (ReLU) activation function.
    * @param x the input value
    * @return the output value of the derivative of the ReLU function
    */

    return ( x>= 0) ? 1.0 : 0.0;
}

std::vector<double> vectRelu(const std::vector<double> x)
{ /**
   * A vectorized version of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input vector
   * @return a vector where each element is the ReLU of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(relu(i));
    return result;
}

std::vector<double> vectReluDerivative(const std::vector<double> x)
{ /**
   * A vectorized version of the derivative of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input vector
   * @return a vector where each element is the derivative of the ReLU function of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(reluDerivative(i));
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double leakyRelu(double x, double alpha = 0.01)
{
    /**
     * The Leaky Rectified Linear Unit (Leaky ReLU) activation function.
     * @param x the input value
     * @param alpha the leak rate, defaults to 0.01
     * @return the output value of the Leaky ReLU function
     */
    return std::max(x*alpha, x);
}

double leakyReluDerivative(double x, double alpha = 0.01)
{
    /**
   * The derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input value
   * @param alpha the leak rate, defaults to 0.01
   * @return the output value of the derivative of the Leaky ReLU function
   */
    return (x >= 0) ? 1.0 : alpha;
}

std::vector<double> vectLeakyRelu(const std::vector<double> x, double alpha = 0.01)
{ /**
   * A vectorized version of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input vector
   * @param alpha the leak rate, defaults to 0.01
   * @return a vector where each element is the Leaky ReLU of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu(i, alpha));
    return result;
}

std::vector<double> vectLeakyReluDerivative(const std::vector<double> x, double alpha = 0.01)
{ /**
   * A vectorized version of the derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input vector
   * @param alpha the leak rate, defaults to 0.01
   * @return a vector where each element is the derivative of the Leaky ReLU function of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyReluDerivative(i, alpha));
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double custom_tanh(double x)
{
    /**
    * The Hyperbolic Tangent (tanh) activation function.
    * @param x the input value
    * @return the output value of the tanh function
    */
    return std::tanh(x);
}

double tanhDerivative(double x)
{
   /**
   * The derivative of the Hyperbolic Tangent (tanh) activation function.
   * @param x the input value
   * @return the output value of the derivative of the tanh function
   */
    return 1 - pow(custom_tanh(x), 2);
}

std::vector<double> vectTanh(const std::vector<double> x)
{ /**
   * A vectorized version of the Hyperbolic Tangent (tanh) activation function.
   * @param x the input vector
   * @return a vector where each element is the tanh of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(custom_tanh(i));
    return result;
}

std::vector<double> vectTanhDerivative(const std::vector<double> x)
{ /**
   * A vectorized version of the derivative of the Hyperbolic Tangent (tanh) activation function.
   * @param x the input vector
   * @return a vector where each element is the derivative of the tanh function of the corresponding element in x
   */
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanhDerivative(i));
    return result;
}
