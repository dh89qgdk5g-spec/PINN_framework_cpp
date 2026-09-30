#pragma once 
#include <Eigen/Dense>
#include <cmath>


//binary cross-entropy loss (BCE)
double BCEloss(Eigen::VectorXd true_label, Eigen::VectorXd pred_prob)
{  /**
   * Binary Cross Entropy Loss
   * @param true_label true labels of the data
   * @param pred_prob predicted probabilities
   * @return binary cross entropy loss
   */
    double sum = 0;
    for (int i = 0; i < pred_prob.size(); i++)
    {
        sum += true_label[i]*log(pred_prob[i]) + (1 - true_label[i]) * log(1 - pred_prob[i]);
    }
    int size = true_label.size();
    double loss = -(1./size)*sum;
    return loss;
}

Eigen::VectorXd BCElossDerivative(Eigen::VectorXd true_label, Eigen::VectorXd pred_prob)
{
   /**
   * Compute derivative of binary cross entropy loss
   * @param true_label true labels of the data
   * @param pred_prob predicted probabilities
   * @return derivative of binary cross entropy loss
   */
   Eigen::VectorXd dev(1);
   dev[0] = (pred_prob[0] - true_label[0]) / ((pred_prob[0])*(1 - pred_prob[0]));
   return dev;
}

//mean-scquared error loss 
double MSEloss(Eigen::VectorXd true_label, Eigen::VectorXd pred)
{
    /**
   * Mean Squared Error Loss
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return mean squared error loss
   */
    double sum = 0;
    for (int i = 0; i < true_label.size(); i++)
    {
        sum += pow(true_label[i] - pred[i] , 2.) ; 
    }
    int size = true_label.size();
    double loss = (1./size) * sum;
    return loss;
}

Eigen::VectorXd MSElossDerivative(Eigen::VectorXd true_label, Eigen::VectorXd pred)
{
    /**
   * Compute derivative of mean squared error loss
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return derivative of mean squared error loss
   */
    Eigen::VectorXd dev = 2.0 * (pred - true_label);
    return dev;
}