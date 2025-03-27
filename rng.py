from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from scipy.stats import chisquare, entropy
import numpy as np
from scipy import stats
import os
import secrets
import random
from Crypto.Random import get_random_bytes
from collections import defaultdict


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
        "is_random": shannon_entropy >= 0.99
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
        mean = np.mean(self.random_bits)
        centered_bits = self.random_bits - mean
        max_lag = min(max_lag, self.bit_len - 1)  # Prevent out-of-bounds
        autocorr_results = {}

        for lag in range(1, max_lag + 1):
            num = np.sum(centered_bits[:self.bit_len-lag] * centered_bits[lag:])
            denom = np.sum(centered_bits[:self.bit_len-lag] ** 2)  # Lag-adjusted denominator
            R_k = num / denom if denom != 0 else 0
            autocorr_results[lag] = float(R_k)

        threshold = 1.96 / np.sqrt(self.bit_len)  # 95% confidence interval
        is_random = all(abs(R_k) < threshold for R_k in autocorr_results.values())

        return {
            "autocorrelation_values": autocorr_results,
            "is_random": is_random
        }

    def cumulative_sums(self):
        """
        return
        ------
        result_forward : dict
            Dictionary containing the p-value and randomness decision for the forward cumulative sums test.
        result_backward : dict
            Dictionary containing the p-value and randomness decision for the backward cumulative sums test.
        """

        threshold = 0.01  # NIST significance level

        # Forward test
        s_forward = np.cumsum(np.array(self.random_bits) * 2 - 1)  # Convert to ±1
        max_forward = np.max(np.abs(s_forward))
        p_value_forward = 1 - stats.norm.cdf(max_forward / np.sqrt(self.bit_len))
        result_forward = {
            "p_value": p_value_forward,
            "is_random": p_value_forward >= threshold
        }

        # Backward test
        s_backward = np.cumsum(np.array(self.random_bits[::-1]) * 2 - 1)  # Cumulative sum of reversed sequence
        max_backward = np.max(np.abs(s_backward))
        p_value_backward = 1 - stats.norm.cdf(max_backward / np.sqrt(self.bit_len))
        result_backward = {
            "p_value": p_value_backward,
            "is_random": p_value_backward >= threshold
        }
        # Store results
        return result_forward, result_backward

    def random_excursions(self):
        """
        Performs the Random Excursions test to detect patterns in the sequence of random bits.

        The test checks the number of cycles of a particular state in the cumulative sum of the bits.
        The test is applied to the states [-4, -3, -2, -1, 1, 2, 3, 4] to detect any non-random behavior.

        Parameters:
        ----------

        Returns:
        -------
        dict
            A dictionary containing:
            - "p_value" : list - A list of p-values for each state.
            - "is_random" : bool - Whether the sequence passes the randomness test 
            (p-value > 0.01 indicates no significant deviation from randomness).
        """
        threshold = 0.01
        multi_pvalue_pass_rate = 0.96
        # Convert bits to a numpy array and transform: 0 -> -1, 1 -> 1
        s = np.cumsum(np.array(self.random_bits, dtype=int) * 2 - 1)
        # Find indices where the cumulative sum is zero
        zero_crossings = np.where(s == 0)[0]
        # Check if there are enough cycles
        if len(zero_crossings) < 2:
            return {"p_value": [], "is_random": False}
        # Number of cycles is the number of segments between zeros
        J = len(zero_crossings) - 1
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        # Initialize counts for each state and k (0 to 5, where 5 means >=5)
        counts = defaultdict(lambda: np.zeros(6, dtype=int))
        # Process each cycle
        for i in range(J):
            start = zero_crossings[i] + 1
            end = zero_crossings[i + 1]
            cycle = s[start:end]  # Values between consecutive zeros
            for x in states:
                v_x = np.sum(cycle == x)  # Number of visits to state x
                k = min(v_x, 5)  # Cap at 5 for k >= 5
                counts[x][k] += 1
        # Expected probabilities π_k(x) from NIST SP 800-22 (for positive x; symmetric for negative)
        pi = {
            1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
            2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
            3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0803],
            4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0732]
        }
        # Compute p-values for each state
        p_values = []
        for x in states:
            pi_x = pi[abs(x)]  # Symmetry: use |x| for probabilities
            f_obs = counts[x]  # Observed frequencies
            f_exp = J * np.array(pi_x)  # Expected frequencies
            # Normalize f_exp to sum to J
            f_exp = f_exp * (J / np.sum(f_exp))
            chi_stat, p_value = stats.chisquare(f_obs, f_exp)
            p_values.append(p_value)
        
        # Determine if the sequence is random
        is_random = (len(p_values) > 0 and 
                    sum(1 for p in p_values if p >= threshold) / len(p_values) >= multi_pvalue_pass_rate)

        return {"p_value": p_values, "is_random": is_random}
    
    def random_excursions_variant(self):
        """
        Performs the Random Excursions Variant test to detect patterns in the sequence of random bits.

        The test checks the number of cycles of a particular state in the cumulative sum of the bits.
        The test is applied to the states [-9, -8, ..., -1, 1, ..., 9] to detect any non-random behavior.

        Parameters:
        ----------

        Returns:
        -------
        dict
            A dictionary containing:
            - "p_value" : list - A list of p-values for each state.
            - "is_random" : bool - Whether the sequence passes the randomness test 
            (p-value > 0.01 indicates no significant deviation from randomness).
        """
        threshold = 0.01
        multi_pvalue_pass_rate = 0.96

        # Transform bits to ±1 and compute cumulative sum
        s = np.cumsum(np.array(self.random_bits, dtype=int) * 2 - 1)
        # Identify zero crossings
        zero_crossings = np.where(s == 0)[0]
        if len(zero_crossings) < 2:
            return {"p_value": [], "is_random": False}
        # Number of cycles
        J = len(zero_crossings) - 1
        states = list(range(-9, 0)) + list(range(1, 10))  # -9 to -1, 1 to 9
        # Count visits per cycle for each state
        counts = defaultdict(lambda: np.zeros(6, dtype=int))  # k = 0 to 5 (5 is >=5)
        for i in range(J):
            start = zero_crossings[i] + 1
            end = zero_crossings[i + 1]
            cycle = s[start:end]
            for x in states:
                v_x = np.sum(cycle == x)
                k = min(v_x, 5)  # Cap at 5 for k >= 5
                counts[x][k] += 1
        # Expected probabilities π_k(x) for |x| = 1 to 9 (NIST approximate values)
        pi_base = {
            1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
            2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
            3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0803],
            4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0732],
            5: [0.9000, 0.0100, 0.0090, 0.0080, 0.0072, 0.0658],
            6: [0.9167, 0.0069, 0.0063, 0.0058, 0.0053, 0.0590],
            7: [0.9286, 0.0051, 0.0047, 0.0043, 0.0039, 0.0534],
            8: [0.9375, 0.0039, 0.0036, 0.0033, 0.0031, 0.0486],
            9: [0.9444, 0.0031, 0.0029, 0.0027, 0.0025, 0.0444]
        }
        
        # Compute p-values
        p_values = []
        for x in states:
            pi_x = pi_base[abs(x)]  # Symmetry for negative states
            f_obs = counts[x]
            f_exp = J * np.array(pi_x)
            # Normalize f_exp to sum to J
            f_exp = f_exp * (J / np.sum(f_exp))
            chi_stat, p_value = stats.chisquare(f_obs, f_exp)
            p_values.append(float(p_value))
        # Randomness decision
        is_random = (len(p_values) > 0 and 
                    sum(1 for p in p_values if p >= threshold) / len(p_values) >= multi_pvalue_pass_rate)

        return {"p_value": p_values, "is_random": is_random}

    def binary_matrix_rank(self):
        """
        Performs the Binary Matrix Rank test to detect non-randomness in the sequence of random bits.

        The test divides the sequence into 32x32 matrices and computes the rank of each matrix.
        The ranks are then compared against expected probabilities to determine randomness.

        Parameters:
        ----------

        Returns:
        -------
        dict
            A dictionary containing:
            - "p_value" : float - The p-value of the test.
            - "is_random" : bool - Whether the sequence passes the randomness test 
            (p-value > 0.01 indicates no significant deviation from randomness).
        """
        threshold = 0.01

        m = 32  # Matrix size (simplified)
        block_size = m * m  # 1024 bits for 32x32 matrix
        num_matrices = self.bit_len // (m * m)
        
        if num_matrices < 1:
            return {"p_value": f"Test was skipped because of insufficient input, Sequence length {self.bit_len} is less than required {block_size} bits", "is_random": 'idk'}
        
        # Form matrices
        bits_array = np.array(self.random_bits[:num_matrices * m * m]).reshape(num_matrices, m, m)
        
        # Binary rank function over GF(2)
        def binary_rank(matrix):
            """
            Compute the rank of a binary matrix over GF(2) using Gaussian elimination.
            """
            mat = matrix.copy().astype(int)
            rows, cols = mat.shape
            rank = 0
            
            for col in range(cols):
                pivot_row = -1
                for r in range(rank, rows):
                    if mat[r, col] == 1:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                
                # Swap rows if necessary
                if pivot_row != rank:
                    mat[[rank, pivot_row]] = mat[[pivot_row, rank]]
                
                # Eliminate column entries below pivot
                for r in range(rows):
                    if r != rank and mat[r, col] == 1:
                        mat[r] ^= mat[rank]  # XOR for addition in GF(2)
                rank += 1
            
            return rank
    
        # Compute ranks
        ranks = [binary_rank(mat) for mat in bits_array]
        
        # Count ranks in three categories
        full_rank_count = sum(1 for r in ranks if r == m)         # r = m
        rank_minus_one_count = sum(1 for r in ranks if r == m - 1) # r = m-1
        lower_rank_count = sum(1 for r in ranks if r <= m - 2)    # r <= m-2
        
        # Observed frequencies
        f_obs = [full_rank_count, rank_minus_one_count, lower_rank_count]
        
        # Expected probabilities (NIST values for Q=32)
        p_full = 0.288788095
        p_minus_one = 0.577576190
        p_lower = 0.133635714
        f_exp = [p_full * num_matrices, p_minus_one * num_matrices, p_lower * num_matrices]
        
        # Chi-square test
        chi_stat, p_value = stats.chisquare(f_obs, f_exp)
        
        return {"p_value": p_value, "is_random": p_value >= threshold}

    def non_overlapping_template_matching(self):
        """
        Performs the Non-overlapping Template Matching test according to NIST SP 800-22.
        This requires bit lentgh to be more than 51,200 bits to be reliable.
        Returns:
            dict: Contains the P-value and a boolean indicating if the sequence is random (P-value >= 0.01).
                  If the sequence is too short, returns a message and 'idk' for is_random.
        """
        # Determine template length and set block size M
        template = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        m = 9
        M = 1024
        
        # Check if the sequence is long enough for at least one block
        if self.bit_len < M:
            return {
                "p_value": f"Test was skipped because of insufficient input, Sequence length {self.bit_len} is less than required {M} bits",
                "is_random": 'idk'
            }
        
        # Calculate number of blocks N
        N = self.bit_len // M
        
        # Divide the sequence into N blocks
        blocks = [self.random_bits[i * M : (i + 1) * M] for i in range(N)]
        
        # Count non-overlapping template matches in each block
        W = []
        for block in blocks:
            count = 0
            i = 0
            while i <= len(block) - m:
                if block[i:i + m] == template:
                    count += 1
                    i += m  # Skip m bits after a match (non-overlapping)
                else:
                    i += 1
            W.append(count)
        
        # Compute theoretical mean and variance
        mu = (M - m + 1) / (2 ** m)
        sigma2 = M * (2 ** (-m) - 2 ** (-2 * m))
        
        # Calculate chi-square statistic
        chi2 = sum((w - mu) ** 2 / sigma2 for w in W)
        
        # Compute P-value using chi-square survival function
        p_value = stats.chi2.sf(chi2, df=N)
        
        # Determine randomness (P-value threshold of 0.01)
        is_random = p_value >= 0.01
        
        # If N < 100, add a warning to the p_value
        if N < 100:
            p_value = f"Warning: Only {N} blocks used (less than 100). P-value may not be reliable. P-value: {p_value:.6f}"
        
        return {"p_value": p_value, "is_random": is_random}

    def overlapping_template_matching(self):
        threshold = 0.01  # NIST significance level
        template = [1] * 9
        template_len = len(template)
        matches_over = 0
        for i in range(self.bit_len - template_len + 1):
            if self.random_bits[i:i+template_len] == template:
                matches_over += 1
        expected = (self.bit_len - template_len + 1) * (0.5 ** template_len)  # Expected number of matches
        p_value_over = stats.poisson.cdf(matches_over, expected)
        return {"p_value": p_value_over, "is_random": p_value_over >= threshold}
    
    def linear_complexity(self):
        """
        
        """
        threshold = 0.01  # NIST significance level
        def berlekamp_massey(bits):
            n = len(bits)
            b = [1] + [0] * n
            c = [1] + [0] * n
            l, m = 0, -1
            for n in range(n):
                d = bits[n] ^ sum(c[i] & bits[n - i] for i in range(1, l + 1)) & 1
                if d:
                    t = c[:]
                    for i in range(n - m, n - m + l + 1):
                        if i < len(c):
                            c[i] ^= b[i - (n - m)]
                    if 2 * l <= n:
                        l, m = n + 1 - l, n
                        b = t
            return l
        M = 500
        blocks = [self.random_bits[i:i+M] for i in range(0, self.bit_len - M + 1, M) if len(self.random_bits[i:i+M]) == M]
        if blocks:
            complexities = [berlekamp_massey(block) for block in blocks]
            mean_complexity = np.mean(complexities)
            expected = M / 2  # Simplified expected complexity
            p_value_lc = stats.norm.cdf(mean_complexity, loc=expected, scale=np.sqrt(M / 4))
            return {"p_value": p_value_lc, "is_random": p_value_lc >= threshold}
        else:
            return {"p_value": 'test was skipped becasue bit len is less than 500', "is_random": "idk"}


    def maurers_universal(self):
        """
        
        """
        threshold = 0.01  # NIST significance level
        L = 6  # Block length
        min_length = 2 ** L * 10
        if self.bit_len >= min_length:
            blocks = [int(''.join(map(str, self.random_bits[i:i+L])), 2) for i in range(0, self.bit_len - L + 1, L)]
            last_seen = {}
            distances = []
            for i, block in enumerate(blocks):
                if block in last_seen:
                    distances.append(i - last_seen[block])
                last_seen[block] = i
            if distances:
                mean_distance = np.mean(distances)
                expected = 2 ** L / 2  # Simplified expected distance
                p_value_univ = stats.norm.cdf(mean_distance, loc=expected, scale=np.sqrt(expected))
                return {"p_value": p_value_univ, "is_random": p_value_univ >= threshold}
            else:
                return {"p_value": "test was skipped due to insufficient input, no repeated blocks found", "is_random": 'idk'}
        else:
            return {"p_value": f"test was skipped because sequence length {self.bit_len} < required {min_length} bits", "is_random": 'idk'}