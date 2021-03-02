# MSc_Special_Topics_Astrodynamics

These files constitute the basis for the assignment of the MSc course "Special Topics in Astrodynamics", serving as an introduction to machine learning in astrodynamics.
Culminated in an individual research paper on the use of machine learning in enabling heteroclinic connections around Lagrange points via solar sail propulsion. 
Following the computation of invariant manifolds in the Earth-Venus restricted three-body problem, a feedforward neural network on TensorFlow was created to minimise 
the transfer time by controlling the sail solar incidence angle.

FILES:

1) Earth_Venus_Manifolds.py 
Computation of lagrange points in Earth-Venus system and invariant manifolds (+ representation)

nn_dataset_generation This is a code which imports the generated manifolds of the Lagrange Points.py file, computed the Euclidean Norm and saves this data in 2 large csv files (training, testing) for ANN training. Efficiently generating data using the special spatial distance function.

ANN_Training ANN architecture implementation, training and graphic visualtisation of outcomes
