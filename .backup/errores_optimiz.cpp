//std::cout << "  dato de entrenamiento: " << input_row << std::endl;
                
                //calcular predicción -------------------------------------------------------------------
                //Eigen::VectorXd prediction = forward_propagation(data_input.row(input_row)); 
                //std::cout << "      ├ predicciónes: " << prediction.transpose() << std::endl; 
                

                //calcular perdida y derivada de peridida -------------------------------------------------
                //double loss = MSEloss(data_output.row(input_row), prediction);
                //total_loss += loss;
                //std::cout << "      ├ Perdida: " << loss << std::endl; 

                //Eigen::VectorXd data_loss_derivative = MSElossDerivative(data_output.row(input_row), prediction);
                //Eigen::VectorXd phys_loss_derivative = Eigen::VectorXd::Ones(data_loss_derivative.rows());
                //std::cout << "      ├ data loss derivative: " << data_loss_derivative.transpose() << std::endl; 
                //std::cout << "      ├ physics loss derivative: " << phys_loss_derivative.transpose() << std::endl; 

                //ejecutar step de optimización ------------------------------------------------------------
                //Layer::backwardOutput losses;
                //losses.error = data_loss_derivative;
                //losses.sensitivity = phys_loss_derivative;

                //backward_propagation(losses, learning_rate);


    void Train2D(const std::vector< Eigen::VectorXd > &X, const std::vector< Eigen::VectorXd> &y, int epochs, double learning_rate ) 
    {
        Layer::backwardOutput prueba;
        prueba.error = Eigen::VectorXd::Ones(1);
        prueba.sensitivity = Eigen::VectorXd::Ones(1);



        for (int epoch = 0; epoch < epochs ; ++epoch) 
        {
            double total_loss = 0. ;

            
            for (size_t i = 0; i < X.size(); ++i) // por cada epoch  recorre los [i] elementos de x, y en cada iteracióon con el elemento [i] hace fprop y bprop
            {
                //forward pass 
                Eigen::VectorXd output = forward_propagation(X[i]);

                // Calcular pérdida
                double loss = BCEloss(y[i], output);
                total_loss += loss;

                Eigen::VectorXd loss_derivative = BCElossDerivative(y[i], output);
                // backward pass 
                backward_propagation(prueba,0.1);
            }
        


            // print loss 
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - loss: " << total_loss / X.size() << std::endl; 
        }
    }

    void Train1D (const Eigen::VectorXd &X, const Eigen::VectorXd &Y, int epochs, double learning_rate)
    {
        

        for (int epoch = 0; epoch < epochs ; ++epoch)
        {
            double total_loss = 0.;
            
            for (int i  = 0; i <X.size(); ++i)
            {
                //f.prop
                Eigen::VectorXd input(1);
                input[0] = X[i];
                
                Eigen::VectorXd output = forward_propagation(input);
                //std::cout << Y[i] - output[0] << std::endl;
                               
                
                // calcular pérdida
                Eigen::VectorXd true_label(1);
                true_label[0] = Y[i];

                double loss = MSEloss(true_label, output);
                total_loss += loss;
               
                //b.prop 
                Eigen::VectorXd data_loss_derivative = MSElossDerivative(true_label, output);
                Eigen::VectorXd physics_loss_derivative = Eigen::VectorXd::Ones(data_loss_derivative.rows());

                Layer::backwardOutput losses;
                losses.error = data_loss_derivative;
                losses.sensitivity = physics_loss_derivative;
                
                backward_propagation(losses, learning_rate);

            }


            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - loss: " << total_loss/X.size()  << std::endl; 
        }
    }
    