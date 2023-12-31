This project implements a planning system for robot navigation as part of the master's lecture ``Intelligent Robotics`` in the winter semester 2023/24 at the Ingolstadt University of Applied Sciences.

### Run the program

The implementation can either be executed locally via the Jupyter notebook `robot-navigation-potential-fields-local.ipynb` or can be started directly via <a target="_blank" href="https://colab.research.google.com/gist/ca-schue/73cff6faf02b6d75d84573625fd89bea/robot-navigation-with-potential-fields.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> without the need for installations. Both are documented in German.

### Theoretical background

A rectangular robot of variable size navigates in a statically predefined occupancy grid with arbitrarily configurable obstacles from a starting point to a destination point.
Collision-free robot navigation is made possible by transforming the robot movement into a three-dimensional configuration space according to the approach of Yunfeng and Chirikjian [1].
The route planning from the starting point to the target point is based on potential fields, the calculation of which was implemented using both attractive and repulsive potentials and the wavefront algorithm. 
The robot is navigated using the gradient descent method in the force fields of the potential fields. 
Various graphical representations visualise the calculations and robot navigation.

The detailed theoretical background of this implementation can be found in the German documentation "paper.pdf".

[1] Yunfeng Wang und Gregory S Chirikjian. „A new potential field method for robot path planning“. In: Proceedings 2000 ICRA. Millennium Conference. IEEE International Conference on Robotics and Automation. Symposia Proceedings (Cat. No. 00CH37065). Bd. 2. 2000, S. 977–982.