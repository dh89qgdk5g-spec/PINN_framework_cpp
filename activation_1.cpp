#pragma once 


#include <cmath>
#include <Eigen/Dense>
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


Eigen::VectorXd vectSigmoid(const Eigen::VectorXd &x)
{
    /**
     * A vectorized version of the sigmoid function.
     * @param x the input vector
     * @return a vector where each element is the sigmoid of the corresponding element in x
     */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = sigmoid(x[i]);
    return result;
}

Eigen::VectorXd vectSigmoidDerivative(const Eigen::VectorXd &x)
{
    /**
     * A vectorized version of the derivative of the sigmoid function.
     * @param x the input vector
     * @return a vector where each element is the derivative of the sigmoid of the corresponding element in x
     */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = sigmoidDerivative(x[i]);
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

Eigen::VectorXd vectRelu(const Eigen::VectorXd &x)
{ /**
   * A vectorized version of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input vector
   * @return a vector where each element is the ReLU of the corresponding element in x
   */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = relu(x[i]);
    return result;
}

Eigen::VectorXd vectReluDerivative(const Eigen::VectorXd &x)
{ /**
   * A vectorized version of the derivative of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input vector
   * @return a vector where each element is the derivative of the ReLU function of the corresponding element in x
   */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = reluDerivative(x[i]);
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

Eigen::VectorXd vectLeakyRelu(const Eigen::VectorXd &x, double alpha = 0.01)
{ /**
   * A vectorized version of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input vector
   * @param alpha the leak rate, defaults to 0.01
   * @return a vector where each element is the Leaky ReLU of the corresponding element in x
   */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = leakyRelu(x[i], alpha);
    return result;
}

Eigen::VectorXd vectLeakyReluDerivative(const Eigen::VectorXd &x, double alpha = 0.01)
{ /**
   * A vectorized version of the derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input vector
   * @param alpha the leak rate, defaults to 0.01
   * @return a vector where each element is the derivative of the Leaky ReLU function of the corresponding element in x
   */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = leakyReluDerivative(x[i], alpha);
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

Eigen::VectorXd vectTanh(const Eigen::VectorXd &x)
{ /**
   * A vectorized version of the Hyperbolic Tangent (tanh) activation function.
   * @param x the input vector
   * @return a vector where each element is the tanh of the corresponding element in x
   */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = custom_tanh(x[i]);
    return result;
}

Eigen::VectorXd vectTanhDerivative(const Eigen::VectorXd &x)
{ /**
   * A vectorized version of the derivative of the Hyperbolic Tangent (tanh) activation function.
   * @param x the input vector
   * @return a vector where each element is the derivative of the tanh function of the corresponding element in x
   */
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
        result[i] = tanhDerivative(x[i]);
    return result;
}