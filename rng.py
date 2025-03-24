from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from scipy.stats import chisquare, entropy
import numpy as np
import os
import secrets
import random
from Crypto.Random import get_random_bytes


class RNG:
    def __init__(self, bit_len, options):
        self.bit_len = bit_len
        self.random_bits = []
        self.options = options
        
    def circuit_rng(self):
        # Clear any previously stored bits
        self.random_bits = []
        
        # Use Qiskit's Aer simulator as the backend
        simulator = Aer.get_backend('aer_simulator')
        
        # Generate one random bit at a time
        for _ in range(self.bit_len):
            # Create a circuit with 1 qubit and 1 classical bit
            qc = QuantumCircuit(1, 1)
            
            # Apply a Hadamard to place the qubit in superposition
            qc.h(0)
            # Measure the qubit into the classical bit
            qc.measure(0, 0)
            # Run the circuit with 1 shot
            job = transpile(qc, simulator)
            job = simulator.run(job)
            result = job.result()
            counts = result.get_counts()
            
            # `counts` will look like {'0': 1} or {'1': 1}
            measured_bit_str = list(counts.keys())[0]  # e.g., '0' or '1'
            measured_bit = int(measured_bit_str)
            self.random_bits.append(measured_bit)
        
        # print("Generated bits:", self.random_bits)
        return self.random_bits

    def os_get_random_bits(self):
        num_bytes = (self.bit_len + 7) // 8  # Calculate the number of bytes needed
        random_bytes = os.urandom(num_bytes)
        random_bits = []

        for byte in random_bytes:
            for i in range(8):
                if len(random_bits) < self.bit_len:
                    random_bits.append((byte >> i) & 1)

        return random_bits

    # PyCryptoDome's get_random_bytes() function
    def generate_secure_random_bits(self):
        num_bytes = (self.bit_len + 7) // 8  # Calculate the number of bytes needed
        random_bytes = get_random_bytes(num_bytes)
        random_bits = []

        for byte in random_bytes:
            for i in range(8):
                if len(random_bits) < self.bit_len:
                    random_bits.append((byte >> i) & 1)

        return random_bits # Truncate to the desired length

    def generate_random_bits(self):
        """
        Generates random bits using a quantum circuit and stores them in the `random_bits` attribute.

        Parameters:
        ----------
        None

        Returns:
        -------
        list
            A list of generated random bits (0s and 1s).
        """
        if self.options == 'qrng':
            self.random_bits = self.circuit_rng()
        elif self.options == 'numpy':
            # rng = np.random.default_rng()
            # random_bits = rng.integers(0, 2, size=self.bit_len)
            self.random_bits = np.random.randint(0, 2, self.bit_len).tolist()
        elif self.options == 'os':
            self.random_bits = self.os_get_random_bits()
        elif self.options == 'secrets':
            rn = secrets.randbits(self.bit_len)
            self.random_bits = [(rn >> i) & 1 for i in range(self.bit_len)]
        elif self.options == 'random':
            rn = random.getrandbits(self.bit_len)
            self.random_bits = [(rn >> i) & 1 for i in range(self.bit_len)]
        elif self.options == 'crypto':
            self.random_bits = self.generate_secure_random_bits()
        else:
            print("Invalid option. Please select one of 'qrng', 'numpy', 'os', 'secrets', or 'random'.") 

        return self.random_bits
    
    def chi_square_test(self):
        """
        Performs the Chi-Square Goodness-of-Fit test to check if the distribution of 0s and 1s 
        in the generated random sequence follows an expected uniform distribution.

        The test compares the observed frequency of 0s and 1s against the expected 
        frequency under the assumption that both occur with equal probability (p=0.5).

        Parameters:
        ----------


        Returns:
        -------
        dict
            A dictionary containing:
            - "chi_square_stat" : float - The computed chi-square statistic.
            - "p_value" : float - The p-value of the test.
            - "is_random" : bool - Whether the sequence passes the randomness test 
            (p-value > 0.05 indicates no significant deviation from uniformity).

        """
        # Ensure that random_bits is not empty
        if not self.random_bits:
            print("No random bits generated yet. Please run `circuit_rng` first.")
            return
        
        # Count occurrences of 0s and 1s
        bit_counts = [self.random_bits.count(0), self.random_bits.count(1)]
        
        # Perform chi-square test
        chi2, p = chisquare(bit_counts)
        
        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p}")
        
        if p > 0.05:
            print("The sequence appears to be random (no evidence against uniformity).")
        else:
            print("The sequence does not appear to be random (evidence of bias).")
        
        return {
            "chi_square_stat": chi2,
            "p_value": p,
            "is_random": p > 0.05
        }

    def shannon_entropy(self):
        """
        Computes the Shannon entropy of a binary sequence to measure its randomness.

        Shannon entropy quantifies the unpredictability of the sequence, with higher 
        values indicating better randomness. The maximum entropy for a binary sequence 
        (equal 0s and 1s) is 1.0. 1.0 indicates unpredictability, while 0.0 indicates
        perfect predictability.

        Parameters:
        ----------

        Returns:
        -------
        dict
            A dictionary containing:
            - "shannon_entropy" : float - The computed entropy value.
            - "is_random" : bool - Whether the entropy value is close to 1 (≥ 0.9 
            indicates a nearly uniform distribution).

        """
        counts = np.bincount(self.random_bits)
        probs = counts / len(self.random_bits)
        shannon_entropy = entropy(probs, base=2) if np.any(probs) else 0.0
    
        return {
        "shannon_entropy": shannon_entropy,
        "is_random": shannon_entropy >= 0.9
        }

    def auto_correlation(self, max_lag=10):
        """
        Computes the autocorrelation of a binary sequence at different lag values 
        to detect any underlying patterns or dependencies.

        A truly random sequence should have near-zero autocorrelation for all 
        lag values, indicating no predictable structure.

        Parameters:
        ----------
        max_lag : int, optional
            The maximum lag to compute autocorrelation for (default is 10).

        Returns:
        -------
        dict
            A dictionary containing:
            - "autocorrelation_values" : dict - A dictionary of autocorrelation 
            coefficients for each lag value.
            - "is_random" : bool - Whether the autocorrelations are close to zero 
            (absolute values < 0.1 indicate randomness).
        """
        n = len(self.random_bits)
        mean = np.mean(self.random_bits)
        autocorr_results = {}

        for lag in range(1, max_lag + 1):
            num = np.sum((self.random_bits[:n-lag] - mean) * (self.random_bits[lag:] - mean))
            denom = np.sum((self.random_bits - mean) ** 2)
            R_k = num / denom if denom != 0 else 0  # Avoid division by zero
            autocorr_results[lag] = R_k

        is_random = all(abs(R_k) < 0.1 for R_k in autocorr_results.values())

        return {
            "autocorrelation_values": autocorr_results,
            "is_random": is_random
        }