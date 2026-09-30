#pragma once
// import dependencies:

#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Dense>


Eigen::VectorXd vInit(double begin, double end, int elements)
{
    /**
    * Generates a vector of evenly spaced values within a specified range, 
    * including both the starting and ending values (similar to NumPy's linspace).
    * 
    * @param begin the starting value of the range
    * @param end the ending value of the range
    * @param elements the total number of elements to generate
    * @return an Eigen::VectorXd containing the generated sequence of values
    */

    Eigen::VectorXd v(elements);
    double step = (end-begin)/(elements-1);
    for (int i = 0; i < elements; i++)
    {
        v[i] = begin + step * i;
    }
    return v;
}

void vprint(const Eigen::VectorXd &v)
{
    /**
    * Prints the contents of a vector to the standard output,
    * formatted as a list enclosed in curly braces (e.g., { 1.0  2.0  3.0 }).
    * @param v a constant reference to an Eigen::VectorXd to be printed
    */

    std::cout << "{";
    for (int i = 0; i < v.size(); i++)
    {
        std::cout <<"  " << v[i]; 
    }
    std::cout << "  }";
    std::cout << std::endl;
}

void mprint(const Eigen::MatrixXd &m)
{
    
    for (int row = 0; row < m.rows(); row ++)
    {
        std::cout << "{";
        for (int col = 0; col < m.cols(); col++)
        {
            std::cout << "  " << m(row, col);
        }
        std::cout << "  }" << std::endl;
    }
    
}

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



