#pragma once
#include <Eigen/Dense>
#include <vector>

#include "1_utils.hpp"
#include "2_activations.hpp"

// Estructura para transportar gradientes de orden superior respecto a salidas, Jacobianos y Hessianos
struct LossGradients
{
    Eigen::VectorXd d_out;              // ∂L/∂(salida)     dim: [n_out]
    Eigen::MatrixXd d_J;                // ∂L/∂(Jacobiano)  dim: [n_out x d]
    std::vector<Eigen::MatrixXd> d_H;   // ∂L/∂(Hessiano)   dim: n_out matrices de [d x d]

    LossGradients() {}
    LossGradients(int n_out, int d)
    {
        d_out = Eigen::VectorXd::Zero(n_out);
        d_J = Eigen::MatrixXd::Zero(n_out, d);
        d_H.resize(n_out, Eigen::MatrixXd::Zero(d, d));
    }
};

class Layer
{
public:
    Eigen::VectorXd input;
    Eigen::VectorXd output;

    // 1st derivatives
    Eigen::MatrixXd localJacobian;
    Eigen::MatrixXd input_Jacobian;
    Eigen::MatrixXd layerJacobian;
    
    // 2nd derivatives
    std::vector<Eigen::MatrixXd> input_Hessian;
    std::vector<Eigen::MatrixXd> layerHessian;
    std::vector<Eigen::MatrixXd> hessian_2; 
    
    Eigen::MatrixXd jacobian_1;

    struct backwardOutput
    {
        Eigen::VectorXd error;       // ∂L/∂x propagador de perdida de datos
        Eigen::VectorXd sensitivity; // ∂u/∂a propadagor de perdida física
    };
    backwardOutput propagations;

    Eigen::MatrixXd W;
    Eigen::VectorXd b;

    virtual ~Layer() = default;
    virtual Eigen::VectorXd forward(const Eigen::VectorXd input_data) = 0;
    virtual void forward_derivatives(const Eigen::MatrixXd &in_J, const std::vector<Eigen::MatrixXd> &in_H) = 0;
    virtual LossGradients backward(const LossGradients &grad) = 0;
    virtual backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) { return propagations; }
    virtual std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_hessian) { return layerHessian; }
    virtual void zero_grad() {}
    virtual void step(double learning_rate) {}
};

class Sigmoid : public Layer 
{
public:
    Eigen::VectorXd derivative_1;

    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        input = input_data;
        output = vectSigmoid(input);
        derivative_1 = vectSigmoidDerivative(input);
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = localJacobian;
        return output;
    }

    void forward_derivatives(const Eigen::MatrixXd &in_J, const std::vector<Eigen::MatrixXd> &in_H) override
    {
        input_Jacobian = in_J;
        input_Hessian = in_H;

        int n = input.size();
        int d = in_J.cols();
        layerJacobian = Eigen::MatrixXd::Zero(n, d);
        layerHessian.resize(n);

        for (int i = 0; i < n; i++)
        {
            double z = input(i);
            double s1 = sigmoidDerivative(z);
            double s2 = sigmoid2Derivative(z);

            layerJacobian.row(i) = s1 * in_J.row(i);

            Eigen::RowVectorXd Ji = in_J.row(i);
            Eigen::MatrixXd Jt_J = Ji.transpose() * Ji;
            layerHessian[i] = s2 * Jt_J + s1 * in_H[i];
        }
        hessian_2 = layerHessian;
    }

    std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_Hessian) override
    {
        return layerHessian;
    }

    LossGradients backward(const LossGradients &grad) override
    {
        LossGradients prev_grad;
        int n = input.size();
        int d = (grad.d_J.cols() > 0) ? grad.d_J.cols() : 
                ((grad.d_H.size() > 0) ? grad.d_H[0].rows() : 0);

        prev_grad.d_out = Eigen::VectorXd::Zero(n);
        if (d > 0)
        {
            prev_grad.d_J = Eigen::MatrixXd::Zero(n, d);
            prev_grad.d_H.resize(n, Eigen::MatrixXd::Zero(d, d));
        }

        for (int i = 0; i < n; i++)
        {
            double z = input(i);
            double s1 = sigmoidDerivative(z);
            double s2 = sigmoid2Derivative(z);
            double s3 = sigmoid3Derivative(z);

            // 1. Propagar Hessiano
            if (grad.d_H.size() > 0)
            {
                prev_grad.d_H[i] = s1 * grad.d_H[i];
            }

            // 2. Propagar Jacobiano
            Eigen::RowVectorXd Jz_i = Eigen::RowVectorXd::Zero(d);
            if (input_Jacobian.size() > 0) Jz_i = input_Jacobian.row(i);

            Eigen::RowVectorXd d_J_i = Eigen::RowVectorXd::Zero(d);
            if (grad.d_J.size() > 0) d_J_i = grad.d_J.row(i);

            Eigen::RowVectorXd term_J = s1 * d_J_i;
            if (grad.d_H.size() > 0 && input_Jacobian.size() > 0)
            {
                Eigen::MatrixXd H_sym = grad.d_H[i] + grad.d_H[i].transpose();
                term_J += s2 * (Jz_i * H_sym);
            }
            if (d > 0)
            {
                prev_grad.d_J.row(i) = term_J;
            }

            // 3. Propagar Salida (z_i)
            double term_z = (grad.d_out.size() > i) ? (grad.d_out(i) * s1) : 0.0;
            if (grad.d_J.size() > 0 && input_Jacobian.size() > 0)
            {
                term_z += s2 * d_J_i.dot(Jz_i);
            }
            if (grad.d_H.size() > 0)
            {
                if (input_Hessian.size() > 0)
                {
                    term_z += s2 * (grad.d_H[i].array() * input_Hessian[i].array()).sum();
                }
                if (input_Jacobian.size() > 0)
                {
                    term_z += s3 * (Jz_i * grad.d_H[i] * Jz_i.transpose())(0, 0);
                }
            }
            prev_grad.d_out(i) = term_z;
        }

        return prev_grad;
    }

    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        propagations.error = derivative_1.cwiseProduct(error);
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
        input  = input_data;
        output = vectRelu(input);
        derivative_1 = vectReluDerivative(input);
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = localJacobian;
        return output;
    }

    void forward_derivatives(const Eigen::MatrixXd &in_J, const std::vector<Eigen::MatrixXd> &in_H) override
    {
        input_Jacobian = in_J;
        input_Hessian = in_H;

        int n = input.size();
        int d = in_J.cols();
        layerJacobian = Eigen::MatrixXd::Zero(n, d);
        layerHessian.resize(n);

        for (int i = 0; i < n; i++)
        {
            double s1 = reluDerivative(input(i));
            layerJacobian.row(i) = s1 * in_J.row(i);
            layerHessian[i] = s1 * in_H[i];
        }
        hessian_2 = layerHessian;
    }

    LossGradients backward(const LossGradients &grad) override
    {
        LossGradients prev_grad;
        int n = input.size();
        int d = (grad.d_J.cols() > 0) ? grad.d_J.cols() : 
                ((grad.d_H.size() > 0) ? grad.d_H[0].rows() : 0);

        prev_grad.d_out = Eigen::VectorXd::Zero(n);
        if (d > 0)
        {
            prev_grad.d_J = Eigen::MatrixXd::Zero(n, d);
            prev_grad.d_H.resize(n, Eigen::MatrixXd::Zero(d, d));
        }

        for (int i = 0; i < n; i++)
        {
            double s1 = reluDerivative(input(i));
            if (grad.d_H.size() > 0) prev_grad.d_H[i] = s1 * grad.d_H[i];
            if (d > 0 && grad.d_J.size() > 0) prev_grad.d_J.row(i) = s1 * grad.d_J.row(i);
            prev_grad.d_out(i) = grad.d_out(i) * s1;
        }
        return prev_grad;
    }

    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        propagations.error = derivative_1.cwiseProduct(error);
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
        input = input_data;
        output = vectLeakyRelu(input, alpha);
        derivative_1 = vectLeakyReluDerivative(input, alpha);
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = localJacobian;
        return output; 
    }

    void forward_derivatives(const Eigen::MatrixXd &in_J, const std::vector<Eigen::MatrixXd> &in_H) override
    {
        input_Jacobian = in_J;
        input_Hessian = in_H;

        int n = input.size();
        int d = in_J.cols();
        layerJacobian = Eigen::MatrixXd::Zero(n, d);
        layerHessian.resize(n);

        for (int i = 0; i < n; i++)
        {
            double s1 = leakyReluDerivative(input(i), alpha);
            layerJacobian.row(i) = s1 * in_J.row(i);
            layerHessian[i] = s1 * in_H[i];
        }
        hessian_2 = layerHessian;
    }

    LossGradients backward(const LossGradients &grad) override
    {
        LossGradients prev_grad;
        int n = input.size();
        int d = (grad.d_J.cols() > 0) ? grad.d_J.cols() : 
                ((grad.d_H.size() > 0) ? grad.d_H[0].rows() : 0);

        prev_grad.d_out = Eigen::VectorXd::Zero(n);
        if (d > 0)
        {
            prev_grad.d_J = Eigen::MatrixXd::Zero(n, d);
            prev_grad.d_H.resize(n, Eigen::MatrixXd::Zero(d, d));
        }

        for (int i = 0; i < n; i++)
        {
            double s1 = leakyReluDerivative(input(i), alpha);
            if (grad.d_H.size() > 0) prev_grad.d_H[i] = s1 * grad.d_H[i];
            if (d > 0 && grad.d_J.size() > 0) prev_grad.d_J.row(i) = s1 * grad.d_J.row(i);
            prev_grad.d_out(i) = grad.d_out(i) * s1;
        }
        return prev_grad;
    }

    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        propagations.error = derivative_1.cwiseProduct(error);
        propagations.sensitivity = derivative_1.cwiseProduct(sensitivity);
        return propagations;
    }
};

class Tanh : public Layer 
{
public:
    Eigen::VectorXd derivative_1;

    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        input = input_data;
        output = vectTanh(input);
        derivative_1 = vectTanhDerivative(input);
        localJacobian = buildJacobian(derivative_1);
        jacobian_1 = localJacobian;
        return output;
    }

    void forward_derivatives(const Eigen::MatrixXd &in_J, const std::vector<Eigen::MatrixXd> &in_H) override
    {
        input_Jacobian = in_J;
        input_Hessian = in_H;

        int n = input.size();
        int d = in_J.cols();
        layerJacobian = Eigen::MatrixXd::Zero(n, d);
        layerHessian.resize(n);

        for (int i = 0; i < n; i++)
        {
            double z = input(i);
            double s1 = tanhDerivative(z);
            double s2 = tanh2Derivative(z);

            layerJacobian.row(i) = s1 * in_J.row(i);

            Eigen::RowVectorXd Ji = in_J.row(i);
            Eigen::MatrixXd Jt_J = Ji.transpose() * Ji;
            layerHessian[i] = s2 * Jt_J + s1 * in_H[i];
        }
        hessian_2 = layerHessian;
    }

    std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_Hessian) override
    {
        return layerHessian;
    }

    LossGradients backward(const LossGradients &grad) override
    {
        LossGradients prev_grad;
        int n = input.size();
        int d = (grad.d_J.cols() > 0) ? grad.d_J.cols() : 
                ((grad.d_H.size() > 0) ? grad.d_H[0].rows() : 0);

        prev_grad.d_out = Eigen::VectorXd::Zero(n);
        if (d > 0)
        {
            prev_grad.d_J = Eigen::MatrixXd::Zero(n, d);
            prev_grad.d_H.resize(n, Eigen::MatrixXd::Zero(d, d));
        }

        for (int i = 0; i < n; i++)
        {
            double z = input(i);
            double s1 = tanhDerivative(z);
            double s2 = tanh2Derivative(z);
            double s3 = tanh3Derivative(z);

            // 1. Propagación de Hessiano:
            if (grad.d_H.size() > 0)
            {
                prev_grad.d_H[i] = s1 * grad.d_H[i];
            }

            // 2. Propagación de Jacobiano:
            Eigen::RowVectorXd Jz_i = Eigen::RowVectorXd::Zero(d);
            if (input_Jacobian.size() > 0) Jz_i = input_Jacobian.row(i);

            Eigen::RowVectorXd d_J_i = Eigen::RowVectorXd::Zero(d);
            if (grad.d_J.size() > 0) d_J_i = grad.d_J.row(i);

            Eigen::RowVectorXd term_J = s1 * d_J_i;
            if (grad.d_H.size() > 0 && input_Jacobian.size() > 0)
            {
                Eigen::MatrixXd H_sym = grad.d_H[i] + grad.d_H[i].transpose();
                term_J += s2 * (Jz_i * H_sym);
            }
            if (d > 0)
            {
                prev_grad.d_J.row(i) = term_J;
            }

            // 3. Propagación de Salida / Activación:
            double term_z = (grad.d_out.size() > i) ? (grad.d_out(i) * s1) : 0.0;
            if (grad.d_J.size() > 0 && input_Jacobian.size() > 0)
            {
                term_z += s2 * d_J_i.dot(Jz_i);
            }
            if (grad.d_H.size() > 0)
            {
                if (input_Hessian.size() > 0)
                {
                    term_z += s2 * (grad.d_H[i].array() * input_Hessian[i].array()).sum();
                }
                if (input_Jacobian.size() > 0)
                {
                    term_z += s3 * (Jz_i * grad.d_H[i] * Jz_i.transpose())(0, 0);
                }
            }
            prev_grad.d_out(i) = term_z;
        }

        return prev_grad;
    }

    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        propagations.error = derivative_1.cwiseProduct(error);
        propagations.sensitivity = derivative_1.cwiseProduct(sensitivity);
        return propagations;
    }
};

class Linear : public Layer 
{
public:
    int input_neuron;
    int output_neuron;

    Eigen::MatrixXd weights; // (num_out, num_in)
    Eigen::VectorXd bias;    // (num_out)

    Eigen::MatrixXd grad_weights;
    Eigen::VectorXd grad_bias;
   
    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitializer(num_out);
        zero_grad();
    }

    void zero_grad() override
    {
        grad_weights = Eigen::MatrixXd::Zero(output_neuron, input_neuron);
        grad_bias = Eigen::VectorXd::Zero(output_neuron);
    }

    void step(double learning_rate) override
    {
        bias -= learning_rate * grad_bias;
        weights -= learning_rate * grad_weights;

        W = weights;
        b = bias;
    }

    std::vector<Eigen::MatrixXd> calculateHessianTensor(std::vector<Eigen::MatrixXd> prev_Hessian) override
    {
        return layerHessian;
    }

    Eigen::VectorXd forward(const Eigen::VectorXd input_data) override
    {
        input = input_data;
        output = weights * input + bias;
        localJacobian = weights;
        jacobian_1 = weights;
        return output;
    }

    void forward_derivatives(const Eigen::MatrixXd &in_J, const std::vector<Eigen::MatrixXd> &in_H) override
    {
        input_Jacobian = in_J;
        localJacobian = weights;
        layerJacobian = weights * in_J;

        input_Hessian = in_H;
        layerHessian = tensorTransform(weights, in_H);
        hessian_2 = layerHessian;
    }

    LossGradients backward(const LossGradients &grad) override
    {
        LossGradients prev_grad;

        // 1. Acumulación de gradientes de parámetros
        Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(output_neuron, input_neuron);
        if (grad.d_out.size() == output_neuron)
        {
            grad_bias += grad.d_out;
            dW += grad.d_out * input.transpose();
            prev_grad.d_out = weights.transpose() * grad.d_out;
        }
        else
        {
            prev_grad.d_out = Eigen::VectorXd::Zero(input_neuron);
        }

        // ∂L/∂W = d_out * input^T + d_J * in_J^T + sum( d_H * in_H )
        if (grad.d_J.size() > 0 && input_Jacobian.size() > 0)
        {
            dW += grad.d_J * input_Jacobian.transpose();
        }
        if (grad.d_H.size() > 0 && input_Hessian.size() > 0)
        {
            for (int i = 0; i < output_neuron; i++)
            {
                for (int j = 0; j < input_neuron; j++)
                {
                    dW(i, j) += (grad.d_H[i].array() * input_Hessian[j].array()).sum();
                }
            }
        }
        grad_weights += dW;

        // 2. Propagación hacia la capa previa
        if (grad.d_J.size() > 0)
        {
            prev_grad.d_J = weights.transpose() * grad.d_J;
        }
        if (grad.d_H.size() > 0)
        {
            prev_grad.d_H = tensorTransform(weights.transpose(), grad.d_H);
        }

        return prev_grad;
    }

    backwardOutput backward(Eigen::VectorXd error, Eigen::VectorXd sensitivity, double learning_rate) override
    {
        Eigen::VectorXd input_error = weights.transpose() * error;
        propagations.error = input_error;
        Eigen::MatrixXd weight_error = error * input.transpose();
        Eigen::VectorXd bias_error = error;

        Eigen::VectorXd input_sensitivity = weights.transpose() * sensitivity;
        propagations.sensitivity = input_sensitivity;

        bias = bias - learning_rate * bias_error;
        weights = weights - learning_rate * weight_error;

        return propagations;
    } 
};
