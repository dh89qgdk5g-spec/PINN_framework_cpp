#include <Eigen/Dense>
#include <iostream>
#include <fstream>





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





int main()
{
    Eigen::MatrixXd matriz = Eigen::MatrixXd::Random(20, 5);
    Eigen::VectorXd Vector = Eigen::VectorXd::Random(4);

    std::ofstream outputfile("parametrs.txt");
    writeMatrix(outputfile, matriz);

    outputfile << "\n";
    writeVector(outputfile, Vector);

    outputfile.close();
    
}