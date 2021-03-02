# MSc_Special_Topics_Astrodynamics

These files constitute the basis for the assignment of the MSc course "Special Topics in Astrodynamics", serving as an introduction to machine learning in astrodynamics.
Culminated in an individual research paper on the use of machine learning in enabling heteroclinic connections around Lagrange points via solar sail propulsion. 
Following the computation of invariant manifolds in the Earth-Venus restricted three-body problem, a feedforward neural network on TensorFlow was created to minimise 
the transfer time by controlling the sail solar incidence angle.

FILES:

1) Earth_Venus_Manifolds.py 
Computation of lagrange points in Earth-Venus system and invariant manifolds (+ representation)

2) ANN_generate_dataset_py
Imports the generated manifolds from Earth_Venus_Manifolds.py , computes the Euclidean Norm and saves this data in 2 large csv files for ANN training + testing.

3) ANN_training.py
ANN architecture implementation, training and graphic visualtisation of outcomes
