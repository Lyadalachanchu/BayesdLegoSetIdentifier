# BayesdLegoSetIdentifier
# BayesdLegoSetIdentifier

## Overview
BayesdLegoSetIdentifier is a project that identifies LEGO sets based on observed LEGO pieces using various probabilistic methods. The project implements methods such as Expectation Maximization (EM), Gibbs Sampling, and Metropolis-Hastings (MH) to estimate the probability distribution over LEGO sets given observed pieces.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/lyadalachanchu20033/BayesdLegoSetIdentifier.git
    cd BayesdLegoSetIdentifier
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Data Preparation
1. Run `data_retriever.py` to fetch LEGO set data and save it as a CSV file:
    ```sh
    python data_retriever.py
    ```

### Running the Main Script
1. Execute the main script with desired parameters:
    ```sh
    python main.py
    ```

### Analyzing Results
1. Use `analyze_results.py` to analyze and plot the results:
    ```sh
    python analyze_results.py
    ```

## Methods
### Expectation Maximization (EM)
- Iteratively estimates the mixture weights until convergence.

### Gibbs Sampling
- Uses Markov Chain Monte Carlo (MCMC) to sample from the posterior distribution.

### Metropolis-Hastings (MH)
- Another MCMC method that adapts the step size to improve convergence.

## Evaluation
The evaluation metrics include precision, recall, F1 score, and average precision (AP). The results are saved in the `results` directory and can be visualized using the provided plotting functions.
