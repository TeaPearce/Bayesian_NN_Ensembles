## Uncertainty in Neural Networks: Approximately Bayesian Ensembling
Code from paper 'Uncertainty in Neural Networks: Approximately Bayesian Ensembling' - https://arxiv.org/abs/1810.05546

Play with the [interactive demo here](https://teapearce.github.io/portfolio/github_io_1_ens/). See how it compares to [dropout here](https://teapearce.github.io/portfolio/github_io_2_drop/).

<p align="center">
<img width="400" src="html_demos/images/html_demo_rec_02.gif"/img>  
</p>

<p align="center">
<img width="400" src="ensemble_intro.png">
</p>

[notebook_examples](notebook_examples) provides lightweight demo code for a Bayesian anchored ensemble in PyTorch / Keras / Tensorflow for regression and classification.

### Main Files
- [script_methods.py](regression/script_methods.py) - this creates figure 4 of the paper, running 1-D regression example for GP, MC dropout, and Anchored Ensemble. Due to compatibility issues with Edward, HMC and VI are now handled elsewhere. 
- [pymc_HMC_VI.py](regression/pymc_HMC_VI.py) - creates VI and HMC plots for figure 4 of paper.
- [main_exp.py](regression/main_exp.py) - runs UCI benchmarking experiments.
- [main_converge.py](regression/main_converge.py) - runs convergence comparison between GP and anchored ensembles as in figure 7 on Boston dataset.
- [script_anch_need.py](regression/script_anch_need.py) - creates regression portion of figure 6, showing why anchored ensembles is a better approximation of the Bayesian posterior than other ensembling methods.
- [2d_class_need_03.py](classification/2d_class_need_03.py) - creates classification portion of figure 6.
- [2d_anch_plots_algorithm.py](2d_plots/2d_anch_plots_algorithm.py) - creates visualisation of the algorithm in figure 2.
- [2d_post_comparison.py](2d_plots/2d_post_comparison.py) - creates visualisation of the posterior comparison made in figure 3 between analytical, RMS and MFVI.


### Supporting Files
- [DataGen.py](regression/DataGen.py) - handles data set generation or read in.
- [hyperparams.py](regression/hyperparams.py) - holds hyperparameters used in UCI benchmarking.
- [utils.py](regression/utils.py) - handles some plotting and support functions.
- [module_gp.py](regression/module_gp.py) - code behind the equivalent gaussian processes.
- [module_NN_ens.py](regression/module_NN_ens.py) - code for anchored ensembling, includes the regularisation around initialisation values (the ‘anchor’ method). Is only set up for single or double layer fully connected NN.

### Requirements
Below are the package versions used in writing the code. 

Python 3.7.0

tensorflow                                                            
'1.12.0'

numpy                                                              
'1.15.3'

matplotlib                                                      
'3.0.1'

sklearn
'0.20.0'

pymc3                              
'3.5'

theano (use with pymc3)
'1.0.3'
