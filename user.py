from qrng import QRNG
import numpy as np

class User():

    def __init__(self, bit_len):
        self.bit_len = bit_len
        self.basis = None
        self.states = list()

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

      