# NN Assisted MPC

## Description

In main.py the NN assisted MPC is implemented. It requires the trained NN within the ./Dumps folder. The NN is implemented as a NN with context whereby information of input/output size, normalization, etc. is stored along the pytorch NN. Before running the main script, one can define the mode, polisher, and sentinel option within the script. After executing the script, a file with all input/output traces is generated which can be used in for comparison.

### Mode

Mode defines what input variables can be changed/varied. Default case is C1 and C2 but also C1, C2, and temp or flow rate are possible. 

### Polisher

Polisher defines the nonlinear optimization algorithm which is used in the PBM-based optimization. The PBM based optimization uses a AD model for the reactor model and the model within the optimization.

### Sentinel

Sentinel (Guard/Police) defines the method which is used to overcome model uncertainties.

- None - No measurement against model uncertainties
- MPC_INTEGRATING_PBM - Disturbance integration which is used as offset in the PBM model

## Run script

- ADJUST:
    + /!\ filename at bottom of file
    + mode, polisher, and sentinel in top of file

- main.py
- Execute form: "./"
