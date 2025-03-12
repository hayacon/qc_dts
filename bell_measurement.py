import numpy as np

# class Bell_measurement:
#     def __init__(  state_1, state_2):
#         self.state_1 = state_1
#         self.state_2 = state_2
#         self.outcome_1 = []
#         self.result = []
#         #check if for variables are the same length, if not return an error
#         assert len(self.state_1) == len(self.state_2), "Variables must have the same length"

###############################################################################
# 1) Define the 50/50 BS transformation (using [1,1;1,-1]/sqrt(2)) for H or V.
###############################################################################
def bs_transform_h(  aH, bH):
    """
    Return a dictionary of (n_cH, n_dH) -> amplitude
    after applying the 50/50 BS matrix:
    a_H^dagger = 1/sqrt(2)*(c_H^dagger + d_H^dagger)
    b_H^dagger = 1/sqrt(2)*(c_H^dagger - d_H^dagger)

    aH,bH ∈ {0,1}, or (1,1) for the two-photon (Hong Ou Mandel) case.
    """
    out_amps = {}

    # case 1: no photons in H
    if (aH + bH) == 0:
        out_amps[(0,0)] = 1.0
        return out_amps
    
    # case 2: exactly one photon in H
    if (aH + bH) == 1:
        # If the photon is from Alice (aH=1):
        #   aH^\dagger -> (cH + dH)/sqrt(2).
        # If from Bob (bH=1):
        #   bH^\dagger -> (cH - dH)/sqrt(2).
        if aH == 1:
            # 1/sqrt(2)*( cH + dH )
            out_amps[(1,0)] = 1/np.sqrt(2)   # c^1 d^0
            out_amps[(0,1)] = 1/np.sqrt(2)  # c^0 d^1
        else:
            # bH=1 => 1/sqrt(2)*( cH - dH )
            out_amps[(1,0)] = 1/np.sqrt(2)
            out_amps[(0,1)] = -1/np.sqrt(2)
        return out_amps

    # case 3: two photons in H (aH=1,bH=1)
    if (aH + bH) == 2:
        # a_H^\dagger b_H^\dagger ->
        #   1/2 [ (cH + dH)(cH - dH) ] = 1/2 [ cH^2 - dH^2 ]
        # => amplitude in cH^2 is +1/2, amplitude in dH^2 is -1/2
        # No cH dH term => perfect bunching, with a relative sign.
        out_amps[(2,0)] =  1/np.sqrt(2)
        out_amps[(0,2)] = -1/np.sqrt(2)
        return out_amps

    raise ValueError("Invalid usage for single-photon MDI-QKD")

def bs_transform_v(  aV, bV):
    """
    Same as bs_transform_h but for the V subspace.
    Identical math, just separate function for code clarity.
    """
    out_amps = {}
    if (aV + bV) == 0:
        out_amps[(0,0)] = 1.0
        return out_amps
    if (aV + bV) == 1:
        if aV == 1:
            # (cV + dV)/sqrt(2)
            out_amps[(1,0)] = 1/np.sqrt(2)
            out_amps[(0,1)] = 1/np.sqrt(2)
        else:
            # (cV - dV)/sqrt(2)
            out_amps[(1,0)] = 1/np.sqrt(2)
            out_amps[(0,1)] = -1/np.sqrt(2)
        return out_amps
    if (aV + bV) == 2:
        out_amps[(2,0)] =  1/np.sqrt(2)
        out_amps[(0,2)] = -1/np.sqrt(2)
        return out_amps
    
    raise ValueError("Invalid usage for single-photon MDI-QKD")

###############################################################################
# 2) Combine H-subspace & V-subspace for single-photon from Alice & Bob
###############################################################################
def two_photon_output_distribution(  alphaH, alphaV, betaH, betaV):
    """
    Build the final state for any alphaH,alphaV,betaH,betaV
    (Alice & Bob's polarization amplitudes).
    Return dict of (n_cH,n_dH,n_cV,n_dV) -> complex amplitude.
    """
    out_dict = {}

    def add_amp(k, amp):
        out_dict[k] = out_dict.get(k,0.0) + amp

    # 4 input combos: aH=1,bH=1 or aH=1,bV=1 or aV=1,bH=1 or aV=1,bV=1
    # Weighted by alphaH*betaH, alphaH*betaV, etc.
    # Then get their BS transforms for H and V separately, combine (tensor).
    
    # 1) aH=1,bH=1
    c1 = alphaH*betaH
    if abs(c1)>1e-15:
        dict_h = bs_transform_h(1,1)
        dict_v = bs_transform_v(0,0)  # no photons in V
        for (ncH, ndH), amp_h in dict_h.items():
            for (ncV, ndV), amp_v in dict_v.items():
                add_amp((ncH, ndH, ncV, ndV), c1*amp_h*amp_v)

    # 2) aH=1,bV=1
    c2 = alphaH*betaV
    if abs(c2)>1e-15:
        dict_h = bs_transform_h(1,0)
        dict_v = bs_transform_v(0,1)
        for (ncH, ndH), amp_h in dict_h.items():
            for (ncV, ndV), amp_v in dict_v.items():
                add_amp((ncH, ndH, ncV, ndV), c2*amp_h*amp_v)

    # 3) aV=1,bH=1
    c3 = alphaV*betaH
    if abs(c3)>1e-15:
        dict_h = bs_transform_h(0,1)
        dict_v = bs_transform_v(1,0)
        for (ncH, ndH), amp_h in dict_h.items():
            for (ncV, ndV), amp_v in dict_v.items():
                add_amp((ncH, ndH, ncV, ndV), c3*amp_h*amp_v)

    # 4) aV=1,bV=1
    c4 = alphaV*betaV
    if abs(c4)>1e-15:
        dict_h = bs_transform_h(0,0)
        dict_v = bs_transform_v(1,1)
        for (ncH, ndH), amp_h in dict_h.items():
            for (ncV, ndV), amp_v in dict_v.items():
                add_amp((ncH, ndH, ncV, ndV), c4*amp_h*amp_v)

    return out_dict

###############################################################################
# 3) Convert final (cH, dH, cV, dV) to detection patterns {D1H,D2H,D1V,D2V}
###############################################################################
def get_detection_distribution(  final_state):
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
    
    for (ncH,ndH,ncV,ndV), amp in final_state.items():
        p = abs(amp)**2
        if p<1e-15: 
            continue
        labels = []
        labels += label('c','H',ncH)
        labels += label('d','H',ndH)
        labels += label('c','V',ncV)
        labels += label('d','V',ndV)
        labels_sorted = sorted(labels)
        outcome = "+".join(labels_sorted)
        detect_probs[outcome] = detect_probs.get(outcome,0.0) + p

    return detect_probs

###############################################################################
# 4) High-level function for polarizations in {H,V,D,A}
###############################################################################

def simulate_two_photon_BSM( a_pol,b_pol):
    def pol_to_amp(pol):
        if pol=='H': return (1.0,0.0)
        if pol=='V': return (0.0,1.0)
        if pol=='D': return (1/np.sqrt(2), 1/np.sqrt(2))
        if pol=='A': return (1/np.sqrt(2),-1/np.sqrt(2))
        raise ValueError("Unknown pol")
    aH,aV = pol_to_amp(a_pol)
    bH,bV = pol_to_amp(b_pol)
    
    fs = two_photon_output_distribution(aH,aV,bH,bV)
    dist = get_detection_distribution(fs)
    return dist

def sample_detection_outcome(prob_dist):
    """
    Takes a dictionary of { outcome: probability } and returns
    one outcome, randomly selected according to those probabilities.
    """
    outcomes = list(prob_dist.keys())
    probs = list(prob_dist.values())
    
    # It's good practice to verify they sum to ~1:
    total = sum(probs)
    if not np.isclose(total, 1.0):
        # If needed, you could normalize them here, or raise an error
        # For example: probs = [p / total for p in probs]
        pass
    
    return np.random.choice(outcomes, p=probs)

def interpret_bsm_outcome(outcome):
    """
    outcome: string like "D1H+D1V", "D2H+D2V", "D1V+D2H", etc.
    
    Returns: one of 
      - "|psi^+>" 
      - "|psi^->" 
      - "no_BSM"
    """

    # In your simulation, you might have stored them in sorted form.
    # For safety, let's define a sorted label:
    parts = outcome.split("+")
    parts_sorted = sorted(parts)  # e.g. ["D1H", "D1V"]
    sorted_outcome = "+".join(parts_sorted)  # "D1H+D1V"

    # Check the two categories:
    # 1) psi^+: "D1H+D1V" or "D2H+D2V" 
    if sorted_outcome in ("D1H+D1V", "D2H+D2V"):
        return "|psi^+>"

    # 2) psi^-: "D1H+D2V" or "D1V+D2H"
    #    sorted forms: "D1H+D2V", "D1V+D2H"
    if sorted_outcome in ("D1H+D2V", "D1V+D2H"):
        return "|psi^->"

    # Otherwise, not one of the desired two Bell states
    return "no_BSM"





if __name__=="__main__":
    test_pairs = [
        ("H","H"), ("H","V"), ("V","V"),
        ("D","D"), ("A","A"), ("D","A"), ("A","D"),
        ("D","H"), ("A","V")
    ]
    for (a,b) in test_pairs:
        result = simulate_two_photon_BSM(a,b)
        print(f"\n---\nAlice={a}, Bob={b}")
        s=0.0
        for k,v in sorted(result.items()):
            if v>1e-12:
                print(f"  {k}: {v:.3f}")
                s+=v
        print(f"  (Sum of probabilities) = {s:.3f}")
        outcome = sample_detection_outcome(result)
        print(f"  Final outcome: {outcome}")
        print(f"  Interpretation: {interpret_bsm_outcome(outcome)}")










