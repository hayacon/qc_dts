from qrng import QRNG


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
                    self.states.append('V')
                else:
                    self.states.append('H')
            else:
                if states_bit[i] == 0:
                    self.states.append('D')
                else:
                    self.states.append('A')
        
        return self.states

      