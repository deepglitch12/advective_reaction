A physics-informed neural network (PINN) is developed to model the transient temperature distribution in a pipe with a constant-velocity fluid flow and localized heating at one end. 
By embedding the governing heat transfer equations into the learning process, the network predicts the spatiotemporal temperature profile of the fluid.
The trained PINN results are then compared with a more traditional ODE solving method such as Runge-Kutta (RK-4).

Files:
- out/make_datasets.py is used to generate datasets for various values of the input of the PINN (such as velocity, alpha, T_in) - These datasets are used to train the PINN with observation data
- src/main_train_adam.py is the main training script. The loss function, optimizer and number of observation, residual and boundary condition datapoints that the model will be trained on can be set here.
- src/validation is used to quickly validate the model for a given scenario and if acceptable, src/results.py is run to store all the results in the 'out' folder.
- src/pinn_model.py contains the MILP model that is used to this experiment
