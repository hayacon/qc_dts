
class Post_processing:
    def __init__(self, bases_1, bases_2, states_1, states_2, result):
        '''
        Parameters
        ----------
        bases_1 : list
            Alice's basis
        bases_2 : list
            Bob's basis
        states_1 : list
            Alice's states in bits (0s and 1s)
        states_2 : list
            Bob's states in bits (0s and 1s)
        result : list
            Bell measurement results
        '''
        self.bases_1 = bases_1
        self.bases_2 = bases_2
        self.states_1 = states_1
        self.states_2 = states_2
        self.sifted_key_1 = []
        self.sifted_key_2 = []
        self.result = result
        self.qber = None
        # self.sifted_result = []

        assert len(self.bases_1) == len(self.bases_2) == len(self.result), "Bases and result must have the same length"

    def sifting(self):
        sifted_outcome = []

        # Make sure all input lists are the same length
        n_rounds = len(relay_outcomes)
        assert all(len(lst) == n_rounds for lst in [alice_bases, bob_bases, alice_bits, bob_bits]), \
        "All input lists must have the same length."


        for i in range(n_rounds):
            if self.result[i] == 'no_BSM':
            # Discard this round
                continue
        
            # if self.bases_1 != self.bases_2:
            #     continue

            if self.bases_1[i] == self.bases_2[i]:
                sifted_outcome.append(self.result[i])

                # Rectilinear basis (Z) => always flip
                if self.bases_1[i] == 'Z':
                    if self.result[i] in ['|psi^->', '|psi^+>']:
                        self.state_2[i] = 1 - self.state_2[i]
                # Diagonal basis (X) => flip only for |psi-|
                elif self.bases_1[i] == 'X':
                    if self.result[i] == '|psi^->':
                        self.state_2[i] = 1 - self.state_2[i]

                self.sifted_key_1.append(self.states_1[i])
                self.sifted_key_2.append(self.states_2[i])
            else:
                continue

        return sifted_outcome
    
    def get_qber(self):
        if len(self.sifted_key_1) != len(self.sifted_key_2):
            raise ValueError("Alice's and Bob's keys must have the same length.")

        if len(self.sifted_key_1) == 0:
            print("No bits to compare. QBER is not defined.")
            return None

        mismatches = sum(a != b for a, b in zip(self.sifted_key_1, self.sifted_key_2))
        self.qber = mismatches / len(self.sifted_key_1)

        return self.qber