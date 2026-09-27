#pragma once
// import dependencies:

#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>


/*
considerrar hacer:

class Matrix
{
private:
    std::size_t rows;
    std::size_t cols;
    std::vector<double> data;
};

*/

std::vector<double> vInit(double begin, double end, int elements)
{
    /**
    * Generates a vector of evenly spaced values within a specified range, 
    * including both the starting and ending values (similar to NumPy's linspace).
    * 
    * @param begin the starting value of the range
    * @param end the ending value of the range
    * @param elements the total number of elements to generate
    * @return a std::vector containing the generated sequence of values
    */

    std::vector<double> v(elements);
    double step = (end-begin)/(elements-1);
    for (int i = 0; i < elements; i++)
    {
        v[i] = begin + step * i;
    }
    return v;
}

void vprint(const std::vector<double> &v)
{
    /**
    * Prints the contents of a double vector to the standard output,
    * formatted as a list enclosed in curly braces (e.g., { 1.0  2.0  3.0 }).
    * @param v a constant reference to the std::vector of doubles to be printed
    */

    std::cout << "{ ";
    for (int i = 0; i < v.size(); i++)
    {
        std::cout << v[i] << "  "; 
    }
    std::cout << '}';
    std::cout << std::endl;
}

double dotProduct(std::vector<double> &v1, std::vector<double> &v2)
{
    /**
     * @brief Computes the dot product of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return The dot product of the two vectors
     */
     
     double result = 0;
     for (int i = 0; i<v1.size(); i++)
     {
        result += v1[i] * v2[i];
     } 
     return result;
}

std::vector<std::vector<double>> buildJacobian(const std::vector<double> &v)
{
    std::vector < std::vector<double> > J(v.size(),std::vector<double> (v.size()))
    
    for(int i = 0; i<v.size(); ++i)
    {
        J[i][i] = v[i];
    }
}

std::vector<double> multiply_Mv(const std::vector< std::vector<double> > &M,const std::vector<double> &v )
{
    std::vector<double> result;
    for(int row = 0; row < M.size(); ++row )
    {
        result.push_back(dotProduct(M[row],v));
    }
    return result;
}

std::vector<std::vector<double>> multiply_MM(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B)
{
    std::vector<std::vector<double>> C(A.size(),std::vector<double> (B[0].size()));
    std::vector<std::vector<double>> B_T = transpose(B);

    for (int row = 0; row<A.size(); ++row)
    {
        for (int col = 0; col<B_T.size(); ++col)
        {
          C[row][col] = dotProduct(A[row],B_T[col]);
        }
    }
    return C;
}




std::vector<double> hadamardProduct(const std::vector<double> &v1, const std::vector<double> &v2)
{
    /**
     * @brief Computes the Hadamard product (element-wise product) of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return A vector containing the element-wise product of the two vectors
     */

    // conviene añadir un mejsaje de error si v1.size() != v2.size()
    std::vector<double> result(v1.size());
    for(int entry = 0; entry<v1.size(); ++entry)
    {
        result[entry] = v1[entry]*v2[entry];
    }
    return result;
}

std::vector<double> scalarVectorMultiplication(std::vector<double> &v,double scalar)
{
  /**
     * @brief Computes the element-wise multiplication of a vector and a scalar
     * @param[in] v The vector to multiply
     * @param[in] scalar The scalar to multiply the vector with
     * @return A new vector with the element-wise multiplication of v and scalar
     */

     std::transform(v.begin(), v.end(), v.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar));
     return v;
}

std::vector<double> subtract(std::vector<double> &v1, std::vector<double> &v2)
{
    /**
     * @brief Computes the element-wise subtraction of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return A new vector with the elementwise subtraction of v1 and v2
     */
     std::vector<double> out;

     std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(out), std::minus<double>());
     return out;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> &m)
{
    /**
     * @brief Computes the transpose of a matrix
     * @param[in] m The matrix to transpose
     * @return The transpose of the matrix
     */

    std::vector<std::vector<double>> trans_vec(m[0].size(), std::vector<double>()); // crea un vector de m[0].size filas y lo llena convectores vacíos 
    for (int i= 0; i< m.size(); i++)
    {
        for (int j = 0; j< m[i].size(); j++){
            if (trans_vec[j].size() != m.size()) // redimensiona el numero de columnas de la fila j 
                trans_vec[j].resize(m.size());
            trans_vec[j][i] = m[i][j]; // rellena las filas y despues las columnas 
        }
    }
    return trans_vec;
}

std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols)
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

     std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));

     for (int i = 0; i < rows; ++i)
     {
        for (int j = 0; j < cols; ++j)
        {
            weights[i][j] = dis(gen);
        }
     }
     return weights;
}

std::vector<double> biasInitializer(int size)
{
    /**
     * @brief Initializes a vector of biases with uniform random weights between -1.0 and 1.0
     * @param[in] size The size of the vector
     * @return A vector of biases with uniform random weights between -1.0 and 1.0
     */

    std::random_device rd;
    std::mt19937 gen(rd() ^std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0,1.0);

    std::vector<double> bias(size);

    for (int i = 0; i<size; ++i)
    {
        bias[i] = dis(gen);
    }
    return bias;
}






