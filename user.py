from qrng import QRNG
import numpy as np

class User():

    def __init__(self, bit_len):
        self.bit_len = bit_len
        self.basis = None
        self.states = []

    def set_basis(self):
        qrng = QRNG(self.bit_len)
        bssis_bits = qrng.circuit_rng()
        self.basis = ['Z' if bit == 0 else 'X' for bit in bssis_bits]
        return self.basis

    def noisy_polarization(self, state, flip_prob=0.05):
        """
        With probability flip_prob, replace pol with a random polarization 
        from the set {H, V, D, A}.
        Otherwise, return the original pol (no change).
        """
        if np.random.rand() < flip_prob:
            # Randomly pick one from the four standard polarizations
            return np.random.choice(["H", "V", "D", "A"])
        else:
            return state

    def state_encoder(self):
        ''' 
        Prepare states in polarization states in Z and X basis {V, H, D, A}
        Note
        ----
        |V> = [1, 0] vertical polarization state
        |H> = [0, 1] horizontal polarization state
        |D> = 1/sqrt(2)(|V> + |H>) = [1, 1] diagonal polarization state
        |A> = 1/sqrt(2)(|V>-|H>) = [1, -1] anti-diagonal polarization state
        '''
        qrng = QRNG(self.bit_len)
        states_bit = qrng.circuit_rng()
        for i in range(len(self.basis)):
            if self.basis[i] == 'Z':
                if states_bit[i] == 0:
                    state = 'V'
                    self.states.append(self.noisy_polarization(state))
                else:
                    state = 'H'
                    self.states.append(self.noisy_polarization(state))
            else:
                if states_bit[i] == 0:
                    state = 'D'
                    self.states.append(self.noisy_polarization(state))
                else:
                    state = 'A'
                    self.states.append(self.noisy_polarization(state))
        
        return self.states, states_bit


class Eve:
    """
    Model Eve intercepting only Alice's qubits in an MDI-QKD setup.
    She chooses a random basis (Z or X) and, if it differs from Alice's, 
    flips the bit with a 50% chance.
    
    This simulates measurement disturbance on one side only.
    """
    def __init__(self, p_alice=0.3):
        """
        p_alice: Probability that Eve intercepts a given qubit from Alice.
        """
        self.p_alice = p_alice

    def intercept(
        self, 
        bases, 
        bits
    ):
        """
        With probability p_alice, Eve intercepts Alice’s qubit 
        and measures in a random basis (Z or X).
        If Eve's basis != Alice's basis, there's a 50% chance 
        that the bit is flipped.
        
        :param bases: list of 'Z'/'X' for each round (Alice's chosen basis).
        :param bits:  list of bits (0/1) for each round (Alice's measurement outcomes).
                            This list is modified in place.
        """
        n = len(bases)
        if len(bits) != n:
            raise ValueError("Mismatch in length: bases vs. bits")

        intercepted_count = 0
        for i in range(n):
            if np.random.rand() < self.p_alice:
                intercepted_count += 1
                eve_basis = np.random.choice(['Z','X'])
                # If Eve's basis != Alice's basis, 50% chance to flip
                if eve_basis != bases[i]:
                    if np.random.rand() < 0.5:
                        bit_int = int(bits[i])
                        bit_int ^= 1
                        bits[i] = bit_int
        
        return intercepted_count

