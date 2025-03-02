from mpqp import QCircuit
from mpqp.gates import H
from mpqp.measures import BasisMeasure
from mpqp.execution import run, IBMDevice
from scipy.stats import chisquare


class QRNG():
    def __init__(self, bit_len):
        self.bit_len = bit_len
        self.random_bits = []
    
    def circuit_rng(self):
        # self.random_bits = []
        for i in range(self.bit_len):
            circuit = QCircuit([H(0), BasisMeasure([0], shots=1)])
            # print(circuit)
            result = run(circuit, IBMDevice.AER_SIMULATOR)
            # print(result._samples[0].index)
            self.random_bits.append(result._samples[0].index)
        print(self.random_bits)
        return self.random_bits
    
    def benchmark(self):
        if not self.random_bits:
            print("No random bits generated yet.")
            return
        
        # Count occurrences of 0s and 1s
        bit_counts = [self.random_bits.count(0), self.random_bits.count(1)]
        
        # Perform chi-square test
        chi2, p = chisquare(bit_counts)
        
        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p}")
        
        if p > 0.05:
            print("The sequence appears to be random.")
        else:
            print("The sequence does not appear to be random.")
        
     

# Example usage
# q = QRNG(10)
# q.circuit_rng()
# q.benchmark()
    