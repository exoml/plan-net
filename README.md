# plan-net-private

An Ensemble of Bayesian Neural Networks for Exoplanetary Atmospheric Retrieval

## Abstract 
Machine learning is now commonly used in many areas of astrophysics, from detecting exoplanets in Kepler transit signals to automatically removing systematic noise in telescopes. These techniques have the ability to both speed-up and automate tasks such as detection, prediction and the generation of complex patterns.
Recent work by \citet{MarquezNeilaEtal2018natureMLRetrieval} recognized the potential of using machine learning algorithms for atmospheric retrieval by implementing a random forest to achieve consistent results with the standard, yet computationally expensive, nested-sampling retrieval method. 
We expand on their approach by presenting a new machine learning model, \texttt{plan-net}, which is specifically designed for atmospheric retrieval.
Our approach, which is based on Bayesian neural networks, achieves a 3 \% improvement over the random forest for the same data set of synthetic transmission spectra. Importantly, we show that designing our machine learning model to explicitly incorporate domain-specific knowledge both improves performance and provides additional insight by inferring the covariance of the retrieved atmospheric parameters.
Whilst the aim of this letter is to illustrate a new machine learning approach for atmospheric retrieval, we highlight that our method is flexible and can be expanded to higher resolution spectra and a larger number of atmospheric parameters.

## A single plan-net model:

![Alt Text](https://github.com/AdamCobb/plan-net-private/blob/master/plan-net_model.png)

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
> M\́{a}rquez-Neila, P., Fisher, C., Sznitman, R., & Heng, K.2018, Nature Astronomy, arXiv:1806.03944

## Contact Information
Adam Cobb: acobb@robots.ox.ac.uk
