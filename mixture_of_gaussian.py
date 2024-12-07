import numpy as np
from sklearn.mixture import GaussianMixture

# randomly generate an N-by-D matrix
np.random.seed(42)
N, D = 1000, 5  
data = np.random.randn(N, D)  

# GMM model to be fitted
num_components = 3  
gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)

# fit GMM to N-by-D matrix
gmm.fit(data)

print("Mean of gassian mixture:")
print(gmm.means_)
print("\ncovariance of gaussian mixture:")
print(gmm.covariances_)

# Given a sample, compute the probability of the sample
sample = np.random.randn(1, D) 
log_prob = gmm.score_samples(sample) 
prob = np.exp(log_prob) 

print(f"\nFor a given sample: {sample}")
print(f"Log likelihood of the sample: {log_prob}")
print(f"Probability of the sample: {prob}")
