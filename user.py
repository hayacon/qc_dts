from qrng import QRNG


class User():

    def __init__(self, bit_len):
        self.bit_len = bit_len
        self.basis = None
        self.states = None

    def set_basis(self):
        qrng = QRNG(self.bit_len)
        bssis_bits = qrng.circuit_rng()
        self.basis = ['Z' if bit == 0 else 'X' for bit in bssis_bits]
        return self.basis
    
    def set_state(self):
        qrng = QRNG(self.bit_len)
        state_bits = qrng.circuit_rng()
        self.states = ['0' if bit == 0 else '1' for bit in state_bits]
        return self.states

        