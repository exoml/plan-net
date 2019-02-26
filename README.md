# plan-net

An Ensemble of Bayesian Neural Networks for Exoplanetary Atmospheric Retrieval

Authors:<br> 
[Adam D. Cobb](http://orcid.org/0000-0003-2868-6983) Department of Engineering Science, University of Oxford<br>
[Michael D. Himes](http://orcid.org/0000-0002-9338-8600) Planetary Science Group, Department of Physics, University of Central Florida<br>
[Frank Soboczenski](http://orcid.org/0000-0001-8185-6094) SPHES, King’s College London<br>
[Simone Zorzan](http://orcid.org/0000-0003-0550-3224) ERIN Department, Luxembourg Institute of Science and Technology<br>
[Molly D. O'Beirne](http://orcid.org/0000-0001-9011-4420) Department of Geology and Environmental Science, University of Pittsburgh<br>
[Atılım Güneş Baydin](http://orcid.org/0000-0001-9854-8100) Department of Computer Science, University of Oxford<br>
[Yarin Gal](https://orcid.org/0000-0002-2733-2078) Department of Computer Science, University of Oxford<br>
[Shawn D. Domagal-Goldman](http://orcid.org/0000-0003-0354-9325) NASA Goddard Space Flight Center, Greenbelt, MD<br>
[Giada N. Arney](http://orcid.org/0000-0001-6285-267X) NASA Goddard Space Flight Center, Greenbelt, MD<br>
[Daniel Angerhausen](http://orcid.org/0000-0001-6138-8633) CSH Fellow, Center for Space and Habitability, University of Bern, Switzerland<br>


## Abstract 
Machine learning is now used in many areas of astrophysics, from detecting exoplanets in Kepler transit signals to removing telescope systematics. 
Recent work demonstrated the potential of using machine learning algorithms for atmospheric retrieval by implementing a random forest to perform retrievals in seconds that are consistent with the traditional, computationally-expensive nested-sampling retrieval method.
We expand upon their approach by presenting a new machine learning model, \texttt{plan-net}, based on an ensemble of Bayesian neural networks
that yields more accurate inferences than the random forest for the same data set of synthetic transmission spectra.
We demonstrate that an ensemble provides greater accuracy and more robust uncertainties than a single model.
In addition to being the first to use Bayesian neural networks 
for atmospheric retrieval, we also introduce a new loss function for Bayesian neural networks 
that learns correlations between the model outputs.
Importantly, we show that designing machine learning models to explicitly incorporate domain-specific knowledge both improves performance and provides additional insight by inferring the covariance of the retrieved atmospheric parameters. We apply \texttt{plan-net} to the Hubble Space Telescope Wide Field Camera 3 transmission spectrum for WASP-12b and retrieve an isothermal temperature and water abundance consistent with the literature. We highlight that our method is flexible and can be expanded to higher-resolution spectra and a larger number of atmospheric parameters.

## A single plan-net model:

![Alt Text](https://github.com/exoml/plan-net/blob/master/plan-net_model.png)

## Reproducing Results
We have created a number of Jupyter notebooks to reproduce our results:

An ensemble of 5 networks should take no longer than 30 minutes to train.

Tested on:
- Ubuntu 18.04, 32GB memory, CPU: Intel Core i7-8700K, GPU: TITAN Xp

## Getting Started

### Requirements
- [Python==3.5](https://www.python.org/getit/)
- [Tensorflow==1.8](https://www.tensorflow.org/)
- [Keras == ?](https://github.com/GPflow/GPflow)
- [Jupyter](http://jupyter.org)

### Installation
1. Clone plan-net and install requirements.
```
cd <installation_path_of_your_choice>
git clone https://github.com/AdamCobb/plan-net
```

2. For the data, clone hela and move data into data folder.
```
git clone https://github.com/exoclime/HELA
cd HELA/example_dataset
cp * <path_to_plan-net>/data/.
```
3. Run notebooks.
```
cd notebooks
jupyter notebook
```

### Running
- Train models by running *plan-net_ensemble_train.ipynb*. One model is already included in *./notebooks/ens_folder_models/* as an example.
- To see the results, run *plan-net_ensemble_train.ipynb*.


## Data
> M\́{a}rquez-Neila, P., Fisher, C., Sznitman, R., & Heng, K. 2018, Nature Astronomy, arXiv:1806.03944

## Contact Information
Adam D. Cobb: `acobb at robots.ox.ac.uk` (Machine Learning questions)<br>
Michael D. Himes: `mhimes at knights.ucf.edu`  (Exoplanetary questions)
