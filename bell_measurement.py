import numpy as np

class Bell_measurement:
    def __init__(self, state_1, state_2, basis_1, basis_2):
        self.state_1 = state_1
        self.state_2 = state_2
        self.outcome_1 = []
        self.outcome_2 = []
        self.result = []
        #check if for variables are the same length, if not return an error
        assert len(self.state_1) == len(self.state_2), "Variables must have the same length"

    def beam_spliter(self, state):
        ''' 50/50 beam splitter matrix '''
        if state == 'H':
            return np.random.choice(['H', 'V'], p=[0.5, 0.5])
        elif state == 'V':
            return np.random.choice(['H', 'V'], p=[0.5, 0.5])
        elif state == 'D':
            return 'H'
        elif state == 'A':
            return 'V'
        else:
            return 'Invalid state'

    def polarization_beam_spliter(self, state, user_side: str):
        ''' 
        Polarization beam splitter matrix 
        user_side: 'Alice' or 'Bob'
        
        '''
        if user_side == 'Alice':
            if state == 'H':
                return 'D2h'
            elif state == 'V':
                return 'D1v'
        elif user_side == 'Bob':
            if state == 'H':
                return 'D1h'
            elif state == 'V':
                return 'D2v'

    def measurement(self):
        ''' 
        perform bell state measurement
        '''
        for i in range(len(self.state_1)):
            state_1 = self.beam_spliter(self.state_1[i])
            state_2 = self.beam_spliter(self.state_2[i])
            outcome_1 = self.polarization_beam_spliter(state_1, 'Alice')
            outcome_2 = self.polarization_beam_spliter(state_2, 'Bob')
            self.outcome_1.append(outcome_1)
            self.outcome_2.append(outcome_2)

        return self.outcome_1, self.outcome_2

    def annunce_result(self):
        ''' 
        Announce the result of the measurement
        ---
        D1h, D1v:|psi+>
        D2h, D2v:|psi+>
        D1h, D2v:|psi->
        D2h, D1v:|psi->
        '''
        for i in range(len(self.outcome_1)):
            if self.outcome_1[i] == 'D1h' and self.outcome_2[i] == 'D1v':
                self.result.append('|psi+>')
            elif self.outcome_1[i] == 'D2h' and self.outcome_2[i] == 'D2v':
                self.result.append('|psi+>')
            elif self.outcome_1[i] == 'D1h' and self.outcome_2[i] == 'D2v':
                self.result.append('|psi->')
            elif self.outcome_1[i] == 'D2h' and self.outcome_2[i] == 'D1v':
                self.result.append('|psi->')
            else:
                self.result.append('Fail')
        return self.result

    












