#include "5_network.hpp"




int main()
{
    // ================================== Set neural network architecture ======================================
    
    // input = {x, y, t}

    // output = {u}
    
    NN network;
    
    network.add(new Linear(3,12));
    network.add(new Tanh);

    network.add(new Linear(12,24));
    network.add(new Tanh);

    network.add(new Linear(24,12));
    network.add(new Tanh);

    network.add(new Linear(12,1));

    
    //  ================================== testing points ====================================
    //domain
    Eigen::Vector2d x_limits(-1,1);
    Eigen::Vector2d y_limits(-1,1);
    Eigen::Vector2d t_limits(0,10);
    
    //initial conditions: ------------------------------------------------------------------------
    int initialConditionPoints = 300;
    Eigen::VectorXd I_x = Eigen::VectorXd::Random(initialConditionPoints);
    Eigen::VectorXd I_y = Eigen::VectorXd::Random(initialConditionPoints);
    Eigen::VectorXd I_t = Eigen::VectorXd::Zero(initialConditionPoints);
    mapVector(x_limits, I_x);
    mapVector(y_limits, I_y);
    Eigen::VectorXd I_u = pow(0.05,(pow(I_x.array(),2)+pow(I_y.array(),2)));

    //estructurar los datos
        // [x_n , y_n , t_n]
        Eigen::MatrixXd IC = Eigen::MatrixXd(initialConditionPoints,3);
        IC.col(0) = I_x;
        IC.col(1) = I_y; 
        
        // [u_n , v_n]
        Eigen::MatrixXd I_output = Eigen::MatrixXd(initialConditionPoints,1);
        I_output.col(0) = I_u;   
    network.IC_trueLabel = I_u;
    
    //boundary testing points:---------------------------------------------------------------
    int boundaryTestingPoints = 200;
    Eigen::VectorXd B_x = Eigen::VectorXd::Random(boundaryTestingPoints);
    Eigen::VectorXd B_y = Eigen::VectorXd::Random(boundaryTestingPoints);
    Eigen::VectorXd B_t = Eigen::VectorXd::Random(boundaryTestingPoints);
    mapVector(t_limits, B_t);

    // Estructurar datos de entrenamiento 
        //[x_n , y_n , t_n]
        Eigen::MatrixXd BC(boundaryTestingPoints*4, 3);
        BC.col(0).segment(0,boundaryTestingPoints) = B_x;
        BC.col(1).segment(0,boundaryTestingPoints).setConstant(y_limits[1]);
        BC.col(2).segment(0,boundaryTestingPoints) = B_t;


        BC.col(0).segment(boundaryTestingPoints,boundaryTestingPoints).setConstant(x_limits[0]); 
        BC.col(1).segment(boundaryTestingPoints,boundaryTestingPoints) = B_y;
        BC.col(2).segment(boundaryTestingPoints,boundaryTestingPoints) = B_t;

        BC.col(0).segment(boundaryTestingPoints*2,boundaryTestingPoints) = B_x;
        BC.col(1).segment(boundaryTestingPoints*2,boundaryTestingPoints).setConstant(y_limits[0]);
        BC.col(2).segment(boundaryTestingPoints*2,boundaryTestingPoints) = B_t;

        BC.col(0).segment(boundaryTestingPoints*3,boundaryTestingPoints).setConstant(x_limits[1]);
        BC.col(1).segment(boundaryTestingPoints*3,boundaryTestingPoints) = B_y;
        BC.col(2).segment(boundaryTestingPoints*3,boundaryTestingPoints) = B_t;
    
        //[u_n, v_n]    
        Eigen::MatrixXd B_output = Eigen::MatrixXd::Zero(BC.rows(),1);
    network.BC_trueLabel = B_output; 
    
    
    // Equation tensting points:------------------------------------------------------------------
    int equationTestingPoints = 100;
    Eigen::VectorXd E_x = Eigen::VectorXd::Random(equationTestingPoints);
    Eigen::VectorXd E_y = Eigen::VectorXd::Random(equationTestingPoints);
    Eigen::VectorXd E_t = Eigen::VectorXd::Random(equationTestingPoints);
    mapVector(x_limits, E_x);
    mapVector(y_limits, E_y);
    mapVector(t_limits, E_t);

        //estructurar datos de entrenamiento
        Eigen::MatrixXd EQ(equationTestingPoints,3);
        EQ.col(0) = E_x;
        EQ.col(1) = E_y;
        EQ.col(2) = E_t;
    
    network.EQ_points = equationTestingPoints;
    
    network.Train(IC,BC,EQ, 5000, 0.00005);

    
    
    
   
    

    


    return 0;
}