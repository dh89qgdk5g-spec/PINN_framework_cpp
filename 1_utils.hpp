#pragma once
// import dependencies:

#include <iostream>
#include <fstream>

#include <random>
#include <chrono>

#include <Eigen/Dense>
#include <vector>


Eigen::MatrixXd buildJacobian(const Eigen::VectorXd &v)
{
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(v.size(), v.size());
    
    for(int i = 0; i < v.size(); ++i)
    {
        J(i,i) = v[i];
    }
    return J;
}

Eigen::MatrixXd uniformWeightInitializer(int rows, int cols)
{
    /**
     * @brief Initializes a matrix with uniform random weights between -1.0 and 1.0
     * @param[in] rows The number of rows in the matrix
     * @param[in] cols The number of columns in the matrix
     * @return A matrix with uniform random weights between -1.0 and 1.0
     */

     std::random_device rd;
     std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
     std::uniform_real_distribution<> dis(-1.0,1.0);

     Eigen::MatrixXd weights(rows, cols);

     for (int i = 0; i < rows; ++i)
     {
        for (int j = 0; j < cols; ++j)
        {
            weights(i,j) = dis(gen);
        }
     }
     return weights;
}

Eigen::VectorXd biasInitializer(int size)
{
    /**
     * @brief Initializes a vector of biases with uniform random weights between -1.0 and 1.0
     * @param[in] size The size of the vector
     * @return A vector of biases with uniform random weights between -1.0 and 1.0
     */

    std::random_device rd;
    std::mt19937 gen(rd() ^std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0,1.0);

    Eigen::VectorXd bias(size);

    for (int i = 0; i<size; ++i)
    {
        bias[i] = dis(gen);
    }
    return bias;
}


std::vector<Eigen::MatrixXd> tensorTransform (Eigen::MatrixXd A, std::vector<Eigen::MatrixXd> T)
{
    /*
    * @brief performs the [A] transformation of the {[T]} tensor, computed similarly as [A]@{x}, 
    * but each entry [A]_ij multiplies a whole matrix [T]
    *
    * @param[in] A the transformation matrix of dimentions [i,j]
    * @param[in] T the tensor to be transformed of dimentions {i} where {[u,v]_i}
    *
    * @return the trasnformed tensor, stores as a vector of i matrixes {[T1],[T2],...[Ti]}
    */
    int c = A.cols(); 
    int r = A.rows();
    std::vector<Eigen::MatrixXd> transformedTensor; 

    for (int i = 0; i < r ; i++)
    {
        Eigen::MatrixXd Hessian = Eigen::MatrixXd::Zero(T[0].rows(),T[0].cols());
        
        for (int j = 0; j < c; j++)
        {
            Hessian += A(i,j) * T[j];
        }
    
        transformedTensor.push_back(Hessian);
    }
    return transformedTensor;
}

void mapVector(Eigen::Vector2d &limits, Eigen::VectorXd &vector)
{
    vector = limits(0) + ((vector.array()+1)/2) * (limits(1)-limits(0));
}

void writeMatrix( std::ofstream &out ,const Eigen::MatrixXd &Matrix)
{
    int rows = Matrix.rows();
    int cols = Matrix.cols();

    for (int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            out << Matrix(i,j) << ",";
        }
        out <<"\n";
    }
}

void writeVector (std::ofstream &out, const Eigen::VectorXd &Vector)
{
    int rows = Vector.rows();

    for (int i = 0; i < rows; i++)
    {
        out << Vector(i) << ",";
    }
    out << "\n";

} 

