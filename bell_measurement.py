import numpy as np

class Bell_measurement:
    def __init__(self, state_1, state_2):
        """
        state_1: list (or tuple) of polarization strings for Alice, e.g. ["H","V","D",...]
        state_2: list of polarization strings for Bob, e.g. ["H","D","V",...]
        
        We assert they have the same length, so we can iterate them in pairs.
        """
        self.state_1 = state_1
        self.state_2 = state_2

        # Check that both lists have the same length:
        assert len(self.state_1) == len(self.state_2), \
            "Alice and Bob must have the same number of states."

        # We'll store results here:
        # a list of dictionaries { 'prob_dist': <dict>, 'sample': <str>, 'interpretation': <str> }
        self.results = []

    # =============================
    # The internal beam-splitter transforms
    # =============================
    def _bs_transform_h(self, aH, bH):
        """
        Return a dictionary of (n_cH, n_dH) -> amplitude
        after applying the 50/50 BS matrix:
           a_H^dagger = (c_H^dagger + d_H^dagger)/sqrt(2)
           b_H^dagger = (c_H^dagger - d_H^dagger)/sqrt(2)
        """
        out_amps = {}

        # case 1: no photons
        if (aH + bH) == 0:
            out_amps[(0,0)] = 1.0
            return out_amps
        
        # case 2: exactly one photon in H
        if (aH + bH) == 1:
            if aH == 1:
                # aH^\dagger => (cH + dH)/sqrt(2)
                out_amps[(1,0)] = 1/np.sqrt(2)
                out_amps[(0,1)] = 1/np.sqrt(2)
            else:
                # bH=1 => (cH - dH)/sqrt(2)
                out_amps[(1,0)] = 1/np.sqrt(2)
                out_amps[(0,1)] = -1/np.sqrt(2)
            return out_amps

        # case 3: two photons in H => Hong-Ou-Mandel
        if (aH + bH) == 2:
            # normalized so that total probability=1 for identical polarizations
            out_amps[(2,0)] =  1/np.sqrt(2)
            out_amps[(0,2)] = -1/np.sqrt(2)
            return out_amps

        raise ValueError("Invalid usage for single-photon MDI-QKD in H-basis")

    def _bs_transform_v(self, aV, bV):
        """
        Same logic as _bs_transform_h, but for the V subspace.
        """
        out_amps = {}
        if (aV + bV) == 0:
            out_amps[(0,0)] = 1.0
            return out_amps
        if (aV + bV) == 1:
            if aV == 1:
                out_amps[(1,0)] = 1/np.sqrt(2)
                out_amps[(0,1)] = 1/np.sqrt(2)
            else:
                out_amps[(1,0)] = 1/np.sqrt(2)
                out_amps[(0,1)] = -1/np.sqrt(2)
            return out_amps
        if (aV + bV) == 2:
            out_amps[(2,0)] =  1/np.sqrt(2)
            out_amps[(0,2)] = -1/np.sqrt(2)
            return out_amps

        raise ValueError("Invalid usage for single-photon MDI-QKD in V-basis")

    # =============================
    # Build the final state from Alice's and Bob's amplitudes
    # =============================
    def _two_photon_output_distribution(self, alphaH, alphaV, betaH, betaV):
        """
        Combine the H-subspace and V-subspace transforms for single photons
        from Alice & Bob. Return a dict of (n_cH,n_dH,n_cV,n_dV) -> complex amplitude.
        """
        out_dict = {}

        def add_amp(k, amp):
            out_dict[k] = out_dict.get(k, 0.0) + amp

        # The 4 possible ways to distribute the photons:
        # 1) aH=1,bH=1
        c1 = alphaH * betaH
        if abs(c1) > 1e-15:
            dict_h = self._bs_transform_h(1,1)
            dict_v = self._bs_transform_v(0,0)
            for (ncH, ndH), amp_h in dict_h.items():
                for (ncV, ndV), amp_v in dict_v.items():
                    add_amp((ncH, ndH, ncV, ndV), c1*amp_h*amp_v)

        # 2) aH=1,bV=1
        c2 = alphaH * betaV
        if abs(c2) > 1e-15:
            dict_h = self._bs_transform_h(1,0)
            dict_v = self._bs_transform_v(0,1)
            for (ncH, ndH), amp_h in dict_h.items():
                for (ncV, ndV), amp_v in dict_v.items():
                    add_amp((ncH, ndH, ncV, ndV), c2*amp_h*amp_v)

        # 3) aV=1,bH=1
        c3 = alphaV * betaH
        if abs(c3) > 1e-15:
            dict_h = self._bs_transform_h(0,1)
            dict_v = self._bs_transform_v(1,0)
            for (ncH, ndH), amp_h in dict_h.items():
                for (ncV, ndV), amp_v in dict_v.items():
                    add_amp((ncH, ndH, ncV, ndV), c3*amp_h*amp_v)

        # 4) aV=1,bV=1
        c4 = alphaV * betaV
        if abs(c4) > 1e-15:
            dict_h = self._bs_transform_h(0,0)
            dict_v = self._bs_transform_v(1,1)
            for (ncH, ndH), amp_h in dict_h.items():
                for (ncV, ndV), amp_v in dict_v.items():
                    add_amp((ncH, ndH, ncV, ndV), c4*amp_h*amp_v)

        return out_dict

    # =============================
    # Convert to detection distribution { "D1H+D2V": prob, ... }
    # =============================
    def _get_detection_distribution(self, final_state):
        detect_probs = {}
        
        def label(mode,pol,count):
            """
            mode in {c,d}, pol in {H,V}, count in {0,1,2}.
            cH => 'D1H', dV => 'D2V', etc.
            """
            if mode=='c' and pol=='H': base="D1H"
            elif mode=='d' and pol=='H': base="D2H"
            elif mode=='c' and pol=='V': base="D1V"
            elif mode=='d' and pol=='V': base="D2V"
            else: base="???"
            return [base]*count
        
        for (ncH, ndH, ncV, ndV), amp in final_state.items():
            p = abs(amp)**2
            if p < 1e-15:
                continue
            labels = []
            labels += label('c','H',ncH)
            labels += label('d','H',ndH)
            labels += label('c','V',ncV)
            labels += label('d','V',ndV)
            labels_sorted = sorted(labels)
            outcome = "+".join(labels_sorted)
            detect_probs[outcome] = detect_probs.get(outcome, 0.0) + p

        return detect_probs

    # =============================
    # Convert a single polarization label {H,V,D,A} into (alphaH, alphaV).
    # =============================
    def _pol_to_amp(self, pol):
        if pol == 'H': return (1.0, 0.0)
        if pol == 'V': return (0.0, 1.0)
        if pol == 'D': # (H+V)/sqrt(2)
            return (1/np.sqrt(2), 1/np.sqrt(2))
        if pol == 'A': # (H-V)/sqrt(2)
            return (1/np.sqrt(2), -1/np.sqrt(2))
        raise ValueError(f"Unknown polarization {pol}")

    # =============================
    # High-level function to run a BSM on a single pair (a_pol,b_pol)
    # =============================
    def simulate_two_photon_BSM(self, a_pol, b_pol):
        aH,aV = self._pol_to_amp(a_pol)
        bH,bV = self._pol_to_amp(b_pol)
        final_state = self._two_photon_output_distribution(aH,aV,bH,bV)
        dist = self._get_detection_distribution(final_state)
        return dist

    # =============================
    # A method to classify outcome => |psi^+>, |psi^->, or no_BSM
    # =============================
    def interpret_bsm_outcome(self, outcome):
        # Sort the labels to handle e.g. "D2V+D1H" equivalently
        parts = outcome.split("+")
        parts_sorted = sorted(parts)
        sorted_outcome = "+".join(parts_sorted)

        # 1) psi^+: "D1H+D1V" or "D2H+D2V"
        if sorted_outcome in ("D1H+D1V", "D2H+D2V"):
            return "|psi^+>"

        # 2) psi^-: "D1H+D2V" or "D1V+D2H"
        if sorted_outcome in ("D1H+D2V", "D1V+D2H"):
            return "|psi^->"

        return "no_BSM"

    # =============================
    # Optionally sample a random detection from the distribution
    # =============================
    def sample_detection_outcome(self, prob_dist):
        outcomes = list(prob_dist.keys())
        probs = list(prob_dist.values())
        total = sum(probs)
        if not np.isclose(total, 1.0):
            # Optionally normalize or raise an error
            pass
        return np.random.choice(outcomes, p=probs)

    # =============================
    # A "run" method to process *all* pairs in state_1, state_2
    # and store results in self.results
    # =============================
    def run_measurements(self):
        """
        Loop over all pairs (state_1[i], state_2[i]) and
        compute the detection probability distribution,
        then store a randomly sampled outcome + interpretation.
        """
        self.results = []  # reset
        for a_pol, b_pol in zip(self.state_1, self.state_2):
            prob_dist = self.simulate_two_photon_BSM(a_pol, b_pol)
            
            # sample a single detection event from prob_dist
            outcome = self.sample_detection_outcome(prob_dist)
            label = self.interpret_bsm_outcome(outcome)

            record = {
                "Alice": a_pol,
                "Bob": b_pol,
                "prob_dist": prob_dist,
                "sample_outcome": outcome,
                "interpretation": label
            }
            self.results.append(record)
        return self.results

    # =============================
    # Helper: print the results
    # =============================
    def print_results(self):
        for i,rec in enumerate(self.results):
            a_pol = rec["Alice"]
            b_pol = rec["Bob"]
            print(f"\nMeasurement #{i}: Alice={a_pol}, Bob={b_pol}")
            dist = rec["prob_dist"]
            # Show nonzero probabilities
            sum_p = 0.0
            for outcome,p in sorted(dist.items(), key=lambda x: x[0]):
                if p>1e-12:
                    print(f"  {outcome}: {p:.3f}")
                    sum_p += p
            print(f"  (Sum of probabilities) = {sum_p:.3f}")
            print(f"  Sampled outcome: {rec['sample_outcome']}")
            print(f"  Interpreted as: {rec['interpretation']}")

    #function to return the interpretation of the results
    def get_interpretation(self):
        interpretation = []
        for i,rec in enumerate(self.results):
            interpretation.append(rec['interpretation'])
        return interpretation


# =========================================================
# Example usage
# =========================================================
if __name__=="__main__":
    # Suppose we have 3 pairs of states
    alice_states = ["H", "D", "V"]
    bob_states   = ["V", "D", "A"]

    bsm = Bell_measurement(alice_states, bob_states)
    bsm.run_measurements()
    bsm.print_results()
