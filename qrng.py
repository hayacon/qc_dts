from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from scipy.stats import chisquare

class QRNG:
    def __init__(self, bit_len):
        self.bit_len = bit_len
        self.random_bits = []
        
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
        
        print("Generated bits:", self.random_bits)
        return self.random_bits
    
    def benchmark(self):
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

# Example usage:
rng = QRNG(bit_len=10)
rng.circuit_rng()
rng.benchmark()


# Example usage
# q = QRNG(10)
# q.circuit_rng()
# q.benchmark()
    