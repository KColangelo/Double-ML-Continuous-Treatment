# Here we simply initialize the Supplment package by importing the modules that
# we will use in the main code.

from .estimation import DDMLCT, NN_DDMLCT, DDMLCT_gps2
from .models import NeuralNet, NeuralNet1_n10000, NeuralNet1_n1000, \
                    NeuralNet2_n10000, NeuralNet2_n1000, NeuralNet1k_n1000, \
                    NeuralNet1k_n10000, NeuralNet1k_emp_app
from .models import NeuralNet1_emp_app, NeuralNet2_emp_app
from .dgp import DGP, DGP2, DGP3, DGP2a
from .file_management import make_dirs
#from .rgrf import regression_forest, regression_forest2





