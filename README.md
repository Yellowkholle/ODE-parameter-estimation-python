ODE-parameter-estimation-python
===============================

Here you can find my codes to find ODE parameters having some noisy observation,
The codes are commented in french (because I am french, but if needed I could translate my comment at english)

I will probably also share my latex work where the theory is presented. My presentation will be at english but the biggest part with all the theory will be at french


I - Lotka-Volterra classical methods.py
---------------------------------------

In this file we can found three classical methods (the gradient descent, the Newton and the Gauss-Newton method).
To try them, in the Lotka-Volterra model, we only to respectively launch :

*  "testMethodeDescenteEnGradientLV()"   &nbsp; &nbsp; #if we want the Gradient descent method  
*  "testMethodesDeNewtonLV()" &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; #if we want to see the Gauss-Newton and Newton method  
*  "testBoiteNoireLV()" &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; #if we want to use a function that already has
*  "test()" &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; #if we want to see all these methods (not the one with named "black box")
      
II - Bayesian approach GP Lotka-Volterra.py
-------------------------------------------

This code is mainly the translation in python of the code https://github.com/odegp/code.  
To try this code with the Lotka-Volterra model, we only need to write in the consol :  

*   "demo_LV()"
