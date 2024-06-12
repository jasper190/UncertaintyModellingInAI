""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Ng Yi Ming
Email: E0486563@u.nus.edu
Student ID: A0211008B
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """

    for seq_idx, trial in enumerate(x_list):
        trialLength = len(trial)

        # Forward run
        alpha = np.zeros([trialLength, n_states])
        scale = np.zeros(trialLength)

        # First step alpha
        alpha[0, :] = pi * scipy.stats.norm.pdf(trial[0], loc=phi['mu'], scale=phi['sigma'])
        scale[0] = np.sum(alpha[0, :])
        alpha[0, :] /= scale[0]

        for t in range(1, trialLength):
            for j in range(n_states):
                alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * scipy.stats.norm.pdf(trial[t], loc=phi['mu'][j], scale=phi['sigma'][j])
            scale[t] = np.sum(alpha[t, :])
            alpha[t, :] /= scale[t]

        # Backward run
        beta = np.ones([trialLength, n_states])
        for t in range(trialLength-2, -1, -1):
            for i in range(n_states):
                beta[t, i] = np.sum(A[i, :] * scipy.stats.norm.pdf(trial[t+1], loc=phi['mu'], scale=phi['sigma']) * beta[t+1, :])
            beta[t, :] /= scale[t+1]

        # Compute gamma and action matrix
        for t in range(trialLength):
            gamma = alpha[t, :] * beta[t, :]
            gamma /= np.sum(gamma)  
            gamma_list[seq_idx][t, :] = gamma

        
        for t in range(trialLength - 1):
            for i in range(n_states):
                for j in range(n_states):
                    xi = alpha[t, i] * A[i, j] * scipy.stats.norm.pdf(trial[t+1], loc=phi['mu'][j], scale=phi['sigma'][j]) * beta[t+1, j]
                    xi_list[seq_idx][t, i, j] = xi
            xi_list[seq_idx][t, :, :] /= np.sum(xi_list[seq_idx][t, :, :])  # Normalize

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    n_sequences = len(x_list)
    pi = np.mean([g[0] for g in gamma_list], axis=0)
    
    # Calc A
    numA = np.zeros((n_states, n_states))
    denomA = np.zeros((n_states, 1))
    
    for i in range(n_sequences):
        numA += np.sum(xi_list[i], axis=0)
        denomA += np.sum(gamma_list[i][:-1], axis=0).reshape(-1, 1)
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-20
    A = numA / (denomA + epsilon)
    
    # Estimate phi
    mu_sum = phi["mu"]
    sigma_sum = phi["sigma"]
    total_weight = 0.0
    
    for i in range(n_sequences):
        sequence = x_list[i]
        gamma = gamma_list[i]
        
        mu_sum += np.dot(gamma.T, sequence)
        sigma_sum += np.dot(gamma.T, sequence**2)
        total_weight += np.sum(gamma, axis=0)
    
    mu = mu_sum / total_weight 
    sigma = np.sqrt(sigma_sum / total_weight - mu**2)
    mu = mu_sum / total_weight 
    sigma = np.sqrt(sigma_sum / total_weight  - mu**2)

    phi = {'mu': mu, 'sigma': sigma}
    
    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    n_states=int(n_states)
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    prev_loglikelihood = -np.inf

    for _ in tqdm(range(max_iters)):
        # E-step
        gamma_list, xi_list = e_step(x_list, pi, A, phi)

        # M-step
        pi, A, phi = m_step(x_list, gamma_list, xi_list)

        #log-likelihood
        loglikelihood = 0
        for i, sequence in enumerate(x_list):
            sequenceLength = len(sequence)
            likelihood = np.zeros(sequenceLength)
            for t in range(sequenceLength):
                likelihood[t] = np.sum(gamma_list[i][t, :] * scipy.stats.norm.pdf(sequence[t], loc=phi['mu'], scale=phi['sigma']))
            loglikelihood += np.sum(np.log(likelihood))

        # Check convergence
        if np.abs(loglikelihood - prev_loglikelihood) < tol:
            break

        prev_loglikelihood = loglikelihood

    return pi, A, phi
