# STA663-waiver-request

# Introduction：
My supporting materials include Python-based projects and relevant Python-based courses with great grades: 
As you can see in the WES report: 
Intro to financial modelling : A
Intro to financial engineering : A
Foundations of financial computing : A
I also have courses : statistical modelling and statistical methods which are all A

This repository contains a selection of my projects. Please check my materials in folders which contain projects description and code. If I can waive STA663, I want to take courses like advanced machine learning and deep learning to improve my ability and my qualificationsalign with the STA 663 syllabus in the following key areas.

# Project 1 ML with finance
## Project Overview

This project applies Python-based data analysis and machine learning techniques to model and analyze financial asset returns. The workflow covers data preprocessing, return computation, feature construction, model training, evaluation, and visualization. The project emphasizes reproducible workflows, computational efficiency, and interpretability of results.

## Methods and Tools

**Programming & Data Handling**
- Python core syntax, control flow, and functions  
- pandas for data manipulation and DataFrame-based analysis  

**Numerical Computing**
- NumPy for numerical computation and array operations  

**Visualization**
- matplotlib and seaborn for exploratory data analysis  

**Machine Learning**
- scikit-learn for model training and evaluation  
- Cross-validation and basic optimization concepts  

## Syllabus Coverage Mapping

- **Weeks 1–2**: Python basics, control flow, data structures  
- **Week 3**: NumPy arrays and numerical computing  
- **Week 4**: pandas DataFrames and data analysis  
- **Week 5**: Data visualization techniques  
- **Weeks 6–7**: scikit-learn, model training, and cross-validation  

# Project 2: Bayesian Survival Analysis with MCMC

## Project Overview

This project focuses on Bayesian survival analysis applied to lung cancer data, with an emphasis on both theoretical foundations and computational implementation. The project studies time-to-event data with censoring and systematically develops survival and hazard functions, non-parametric, semi-parametric, and parametric survival models. Bayesian inference is then introduced to quantify uncertainty in model parameters using prior distributions and posterior analysis.

In addition to classical survival methods, the project implements Markov Chain Monte Carlo (MCMC) techniques, including the Metropolis–Hastings algorithm, to approximate posterior distributions when closed-form solutions are unavailable. The project integrates mathematical derivations, statistical modeling, and computational simulation to provide a comprehensive treatment of Bayesian survival analysis.

## Methods and Tools

**Survival Analysis**
- Survival and hazard functions
- Censored data handling
- Kaplan–Meier estimator and log-rank test
- Cox proportional hazards model
- Parametric survival models (Exponential, Weibull, Gompertz, Log-logistic)
- Model comparison using AIC and likelihood ratio tests

**Bayesian Inference**
- Bayes’ theorem and posterior inference
- Conjugate priors
- Bayesian parameter estimation
- Credible intervals and uncertainty quantification

**MCMC and Computation**
- Markov Chain Monte Carlo (MCMC)
- Metropolis–Hastings algorithm
- Posterior sampling and convergence diagnostics
- Trace plots and posterior density visualization

## Syllabus Coverage Mapping

- **Weeks 6–7**: Statistical modeling and likelihood-based inference  
- **Week 8**: Optimization concepts and likelihood maximization  
- **Weeks 13–14**: MCMC methods, Bayesian inference, and samplers  

# Project 3: Investment Strategy Optimization Using Genetic Algorithms

## Project Overview

This project studies optimal investment, production, transportation, and pricing strategies for bronze products under resource constraints and network structures. The problem is formulated as a profit maximization task over a multi-period horizon, where demand depends on population size and pricing, and transportation costs depend on network connectivity between locations.

The project models the system as a network optimization problem and incorporates production capacity constraints, investment costs, and logistics costs. To solve the resulting high-dimensional, non-convex optimization problem, evolutionary optimization methods are employed. In particular, a hybrid Whale Optimization Algorithm combined with Extreme Learning Machine (WOA-ELM) is used to search for optimal investment strategies under different parameter settings. An extended version of the model further incorporates strategic interactions and conflict dynamics between competing regions.

[The code is at the very end of the article]


## Methods and Tools

**Mathematical Modeling**
- Network-based optimization modeling
- Profit maximization under budget and capacity constraints
- Multi-period decision modeling

**Optimization & Machine Learning**
- Genetic and evolutionary optimization methods
- Whale Optimization Algorithm (WOA)
- Extreme Learning Machine (ELM)
- Hybrid WOA–ELM optimization framework

**Data Processing & Analysis**
- Data normalization and preprocessing
- Constraint handling and feasibility checks
- Simulation-based strategy evaluation


## Syllabus Coverage Mapping

- **Weeks 2–3**: Data structures and numerical computation  
- **Weeks 7–8**: Optimization methods and objective function design  
- **Weeks 9–10**: Stochastic and heuristic optimization algorithms  

# Project 4: Clustering and Neural Network–Based Time Series Modeling

## Project Overview

This project investigates time series modeling using neural networks combined with representation learning and similarity-based aggregation. The task focuses on multivariate time series prediction, where temporal dependencies and cross-variable relationships are jointly modeled. Sliding-window techniques are used to transform raw sequential data into supervised learning samples.

A neural architecture based on recurrent neural networks (GRU/LSTM) is implemented to extract temporal representations. To further enhance predictive performance, the model incorporates similarity-based aggregation mechanisms, where hidden representations from historical data are retrieved and combined using cosine similarity. This design integrates ideas from clustering, representation learning, and neural networks.

The project emphasizes end-to-end deep learning workflows, including dataset construction, model design, training with early stopping, and evaluation using error and correlation metrics.

## Methods and Tools

**Data Processing**
- Sliding-window construction for time series data
- Train/validation/test splitting
- Feature-wise normalization

**Neural Networks**
- Recurrent neural networks (GRU / LSTM)
- Multi-layer feedforward neural networks
- Nonlinear activation functions

**Representation Learning & Similarity**
- Hidden state extraction
- Cosine similarity computation
- Similarity-based neighbor aggregation (implicit clustering)

**Optimization & Training**
- PyTorch training pipeline
- Adam optimizer
- Gradient clipping and early stopping

**Evaluation**
- Relative Squared Error (RSE)
- Correlation (CORR) metrics

## Syllabus Coverage Mapping

- **Weeks 3–4**: NumPy, pandas, and data preparation  
- **Weeks 7–8**: Optimization concepts and model training  
- **Weeks 10–11**: PyTorch, neural networks, and autograd  
- **Week 11**: GPU-aware model execution and deep learning workflows  

# Project 5: High-Accuracy Spiking Neural Network for MNIST Classification

## Project Overview

This project implements a high-accuracy Spiking Neural Network (SNN) for handwritten digit classification on the MNIST dataset. The model is designed using biologically inspired spiking neurons and temporal dynamics, where information is accumulated over multiple discrete time steps. The project focuses on achieving competitive classification accuracy while preserving the event-driven nature of spiking neural computation.

A convolutional SNN architecture is constructed using leaky integrate-and-fire (LIF) neurons, combined with temporal spike accumulation and surrogate gradient learning. The model is trained and evaluated using GPU acceleration, achieving over 98% test accuracy within a small number of training epochs. In addition to performance evaluation, the project includes visualization of training dynamics and spike activity patterns.

## Methods and Tools

**Spiking Neural Networks**
- Leaky Integrate-and-Fire (LIF) neuron models
- Temporal spike accumulation over multiple time steps
- Surrogate gradient backpropagation

**Deep Learning**
- Convolutional neural network architectures
- Cross-entropy loss and Adam optimization

**Computation**
- PyTorch deep learning framework
- GPU-accelerated training
- Training and evaluation visualization

## Syllabus Coverage Mapping

- **Weeks 10–11**: Neural networks and PyTorch fundamentals  
- **Week 11**: GPU-based training and deep learning workflows  

# Project 6: Spiking Neural Network for CIFAR-10 Image Classification

## Project Overview

This project extends spiking neural network modeling to a more challenging image classification task using the CIFAR-10 dataset. Compared to MNIST, CIFAR-10 introduces higher-dimensional inputs and greater visual complexity, requiring deeper architectures and more robust training strategies.

A lightweight convolutional SNN is implemented with integrate-and-fire (IF) neurons, temporal spike-based computation, and surrogate gradient learning. The model incorporates data augmentation, learning-rate scheduling, and GPU-optimized training to improve generalization performance. Evaluation includes accuracy tracking, loss analysis, confusion matrices, and spike activity visualization.

The project emphasizes scalable SNN design and demonstrates how spiking models can be trained effectively on realistic image datasets using modern deep learning infrastructure.

## Methods and Tools

**Spiking Neural Networks**
- Integrate-and-Fire (IF) neuron models
- Time-step–based spike accumulation
- Differentiable spike approximation

**Deep Learning**
- Convolutional architectures for image classification
- Data augmentation and regularization
- Learning rate scheduling

**Computation and Evaluation**
- PyTorch with GPU acceleration
- Accuracy, loss, and confusion matrix analysis
- Spike activity visualization

## Syllabus Coverage Mapping

- **Weeks 10–11**: PyTorch, neural networks, and autograd  
- **Week 11**: GPU execution and deep learning optimization  

# Mathematical Contest in Modeling (MCM/ICM) 2024 – Problem C

## Project Overview

This project was completed as part of the Mathematical Contest in Modeling / Interdisciplinary Contest in Modeling (MCM/ICM) 2024, Problem C. The project focuses on developing a mathematical and computational framework to analyze a real-world system involving multi-factor decision-making, optimization, and data-driven evaluation.

The work integrates mathematical modeling, statistical analysis, and algorithmic implementation to address the problem objectives. Multiple submodels are constructed to capture different components of the system, and quantitative metrics are defined to evaluate model performance. Numerical simulation and sensitivity analysis are used to assess robustness and interpretability of the proposed solutions.

All models are implemented computationally to support scenario analysis and comparative evaluation.


## Modeling and Methods

**Mathematical Modeling**
- Assumption-driven model formulation
- Variable definition and constraint construction
- Objective function design and trade-off analysis

**Statistical and Numerical Analysis**
- Data preprocessing and normalization
- Descriptive statistics and trend analysis
- Sensitivity analysis under parameter variation

**Optimization and Algorithms**
- Multi-objective optimization strategies
- Scenario-based decision evaluation
- Algorithmic implementation of model solutions

**Computation**
- Python-based numerical simulation
- Modular implementation across multiple analytical stages
- Visualization for result interpretation

## Syllabus Coverage Mapping

- **Weeks 1–2**: Python fundamentals and data structures  
- **Weeks 3–4**: Numerical computation and data analysis  
- **Weeks 7–8**: Optimization modeling and objective functions  
- **Weeks 9–10**: Algorithmic modeling and simulation  

