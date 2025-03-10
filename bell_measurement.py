import numpy as np


class Bell_measurement:
    def __init__(self, state_1, state_2, basis_1, basis_2):
        self.state_1 = state_1
        self.state_2 = state_2
        #check if for variables are the same length, if not return an error
        assert len(self.state_1) == len(self.state_2), "Variables must have the same length"

    def beam_spliter(self, state):
        ''' 50/50 beam splitter matrix '''
        pass

    def polarization_beam_spliter(self, state):
        if state == 'H':
            return 'H'
        elif state == 'V':
            return 'V'

    def detector(self):
        pass
    












