#include <Eigen/Dense>
#include <iostream>


int main()
{
    using namespace std;



    Eigen::MatrixXd v1v2;


    Eigen::VectorXd v1 = Eigen::VectorXd::Random(3);
    Eigen::VectorXd v2 = Eigen::VectorXd::Random(3);

    v1v2.resize(3,2);
    v1v2.col(0) = v1;
    v1v2.col(1) = v2;



    cout << "v1: "<< endl << v1.array()+1 << endl;
    
    //cout << "v2: "<< endl << v2 << endl;
    //cout << "v1v2: " << endl << v1v2 << endl;

    return 0;
}