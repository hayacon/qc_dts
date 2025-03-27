import numpy as np
import hashlib
import math

#######################################################
# Alice's Side
#######################################################
class AlicePostProcessing:
    def __init__(self, alice_bases, alice_bits):
        self.alice_bases = alice_bases
        self.alice_bits  = alice_bits
        self.sifted_key  = []  # after sifting

    def sifting(self, bob_bases, relay_outcomes):
        """
        Keep only rounds where relay outcome != 'no_BSM'
        and alice_bases[i] == bob_bases[i].
        (No flips needed for Alice in standard MDI-QKD.)
        """
        self.sifted_key = []
        n = len(self.alice_bases)
        for i in range(n):
            if relay_outcomes[i] == 'no_BSM':
                continue
            if self.alice_bases[i] == bob_bases[i]:
                self.sifted_key.append(self.alice_bits[i])
        return self.sifted_key, len(self.sifted_key)

    def reveal_subset_bits(self, indices):
        """
        Alice reveals ONLY the bits at 'indices' from her sifted key.
        Returns a dictionary: { index: bit_val, ... }
        She does NOT send the entire key, just the subset.
        """
        revealed = {}
        for i in indices:
            if 0 <= i < len(self.sifted_key):
                revealed[i] = self.sifted_key[i]
        return revealed

    def compute_parity(self, block_indices):
        """
        Alice computes the parity (sum mod 2) of her bits 
        over the block defined by 'block_indices' 
        and reveals ONLY the parity, not individual bits.
        """
        total = 0
        for i in block_indices:
            if 0 <= i < len(self.sifted_key):
                total += self.sifted_key[i]
        return total % 2

    def remove_indices(self, indices_to_remove):
        """
        Used to remove certain revealed bits from the final key 
        to ensure they're not reused for encryption.
        """
        # Convert to set for faster lookups
        idx_set = set(indices_to_remove)
        new_key = []
        for i, bit in enumerate(self.sifted_key):
            if i not in idx_set:
                new_key.append(bit)
        self.sifted_key = new_key
        return self.sifted_key

    def binary_entropy(self, p):
        """Compute binary entropy of p."""
        if p <= 0 or p >= 1:
            return 0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def privacy_amplification(self, qber, salt=b"shared-seed", epsilon=0.001):
        """Perform simplified privacy amplification."""
        n = len(self.sifted_key)
        if n == 0:
            return []

        # Compute binary entropy and leakage
        h_Q = self.binary_entropy(qber)  
        leak_EC = 1.2 * n * h_Q  # Information leaked during error correction
        security_margin = 2 * math.log2(1 / epsilon)  # Security buffer

        # Calculate final key length
        final_length = max(0, int(n - leak_EC - security_margin))

        # Convert sifted key to string
        key_str = "".join(str(bit) for bit in self.sifted_key)
        combined_str = key_str + salt.decode('utf-8', errors='ignore')

        # Hash with SHA-256
        h = hashlib.sha256(combined_str.encode()).hexdigest()
        hash_bin = bin(int(h, 16))[2:].zfill(256)  # Convert to binary string

        # Take first final_length bits
        final_bits_str = hash_bin[:final_length]
        final_bits = [int(b) for b in final_bits_str]

        return final_bits

#######################################################
# Bob's Side
#######################################################
class BobPostProcessing:
    def __init__(self, bob_bases, bob_bits):
        self.bob_bases = bob_bases
        self.bob_bits  = bob_bits
        self.qber_estimate = 0.0
        self.sifted_key = []

    def sifting(self, alice_bases, relay_outcomes):
        """
        Keep only rounds where relay outcome != 'no_BSM'
        and bob_bases[i] == alice_bases[i].
        Then apply standard MDI bit-flips for 'psi^-' etc.
        """
        self.sifted_key = []
        n = len(self.bob_bases)
        for i in range(n):
            if relay_outcomes[i] == 'no_BSM':
                continue
            if self.bob_bases[i] == alice_bases[i]:
                # bit-flip rule for MDI:
                outcome = relay_outcomes[i]
                if self.bob_bases[i] == 'Z':
                    if outcome in ['|psi^->', '|psi^+>']:
                        self.bob_bits[i] = 1 - self.bob_bits[i]
                elif self.bob_bases[i] == 'X':
                    if outcome == '|psi^->':
                        self.bob_bits[i] = 1 - self.bob_bits[i]

                self.sifted_key.append(self.bob_bits[i])
        return self.sifted_key, len(self.sifted_key)

    def estimate_qber_sample(self, alice_revealed):
        """
        Bob receives a dictionary of { index: bit_val } from Alice 
        for a small subset of positions. He compares 
        with his own bits at those positions => QBER in that sample.
        
        Return (qber_estimate, list_of_indices_that_were_compared).
        """
        if not alice_revealed:
            return None, []

        n_mismatch = 0
        for idx, a_bit in alice_revealed.items():
            if idx < len(self.sifted_key):
                if self.sifted_key[idx] != a_bit:
                    n_mismatch += 1

        self.qber_estimate = n_mismatch / len(alice_revealed)
        indices = sorted(alice_revealed.keys())
        return self.qber_estimate, indices

    def remove_indices(self, indices_to_remove):
        """
        Remove the bits at 'indices_to_remove' from Bob's key
        to ensure they are not used as final key bits.
        """
        idx_set = set(indices_to_remove)
        new_key = []
        for i, bit in enumerate(self.sifted_key):
            if i not in idx_set:
                new_key.append(bit)
        self.sifted_key = new_key
        return self.sifted_key

    ##############################################################################
    # CASCADE-like multi-pass error correction
    ##############################################################################
    def cascade_pass(self, alice_obj, block_size=4, perm=None):
        """
        One pass of CASCADE-like correction:
          - If perm is not None, we reorder Bob's bits by that permutation to do block checks,
            then revert them at the end.
          - We break the (permuted) key into blocks of 'block_size'.
          - For each block, we compare parity with Alice:
              if mismatch => run a binary search to find & flip the erroneous bit
        """
        n = len(self.sifted_key)
        if n == 0: 
            return

        # If we have a permutation, apply it
        if perm is not None:
            # We'll keep track of the permuted array and revert later
            perm_key = [self.sifted_key[p] for p in perm]
        else:
            perm = list(range(n))
            perm_key = self.sifted_key[:]

        block_count = (n + block_size - 1)//block_size
        for b_idx in range(block_count):
            start = b_idx*block_size
            end   = min(start+block_size, n)
            block_indices = list(range(start, end))

            # Map block_indices to original key indices via inverse of permutation
            real_indices = [perm[i] for i in block_indices]

            # Bob's parity
            bob_par = 0
            for idx in block_indices:
                bob_par ^= perm_key[idx]

            # Ask Alice for parity (using real_indices in her key)
            alice_par = alice_obj.compute_parity(real_indices)

            if bob_par != alice_par:
                # There's exactly 1 error in this block, do a binary search
                self.binary_search_flip(alice_obj, perm_key, block_indices, perm)

        # After finishing pass, revert the permuted array to the original order
        if perm is not None:
            # We have perm_key, which is the updated version in permuted order
            # We need to invert perm
            inverse_perm = [0]*n
            for i,p in enumerate(perm):
                inverse_perm[p] = i
            # Now apply inverse_perm
            new_key = [0]*n
            for i in range(n):
                new_key[i] = perm_key[inverse_perm[i]]
            self.sifted_key = new_key
        return self.sifted_key

    def binary_search_flip(self, alice_obj, perm_key, block_indices, perm):
        """
        Recursively find the single erroneous bit in 'block_indices'.
        We'll do a standard 1-bit error search approach:
          - If there's only 1 bit => flip it
          - Otherwise, check the parity of the left half, compare with Alice,
            then pick which half has the error
        """
        if len(block_indices) == 1:
            # Flip this bit
            idx = block_indices[0]
            perm_key[idx] ^= 1
            return

        mid = len(block_indices)//2
        left_part  = block_indices[:mid]
        right_part = block_indices[mid:]

        # Compute Bob's parity of left_part
        bob_left_par = 0
        for i in left_part:
            bob_left_par ^= perm_key[i]

        # Map left_part to real indices
        left_real = [perm[i] for i in left_part]

        # Compare to Alice's parity
        alice_left_par = alice_obj.compute_parity(left_real)

        if bob_left_par == alice_left_par:
            # error must be in the right half
            self.binary_search_flip(alice_obj, perm_key, right_part, perm)
        else:
            # error is in the left half
            self.binary_search_flip(alice_obj, perm_key, left_part, perm)

    def cascade_ec(self, alice_obj, pass_block_sizes=[4,4], randomize_passes=True):
        """
        A multi-pass error-correction approach:
          pass_block_sizes: list of block sizes for each pass
          randomize_passes: if True, for each pass after the first, 
                            we create a random permutation to reorder bits
                            so errors not found in pass 1 might appear in different blocks in pass 2.
        """
        n = len(self.sifted_key)
        for pass_idx, bsize in enumerate(pass_block_sizes, start=1):
            # maybe shuffle the bit positions for subsequent passes
            if randomize_passes and pass_idx > 1:
                perm = np.random.permutation(n).tolist()
            else:
                perm = None
            self.cascade_pass(alice_obj, block_size=bsize, perm=perm)
        # Additional passes can be done if QBER is still high.

    def binary_entropy(self, p):
        """Compute binary entropy of p."""
        if p <= 0 or p >= 1:
            return 0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def privacy_amplification(self, salt=b"shared-seed", epsilon=0.001):
        """Perform simplified privacy amplification."""
        n = len(self.sifted_key)
        if n == 0:
            return []

        # Compute binary entropy and leakage
        h_Q = self.binary_entropy(self.qber_estimate)  
        leak_EC = 1.2 * n * h_Q  # Information leaked during error correction
        security_margin = 2 * math.log2(1 / epsilon)  # Security buffer

        # Calculate final key length
        final_length = max(0, int(n - leak_EC - security_margin))

        # Convert sifted key to string
        key_str = "".join(str(bit) for bit in self.sifted_key)
        combined_str = key_str + salt.decode('utf-8', errors='ignore')

        # Hash with SHA-256
        h = hashlib.sha256(combined_str.encode()).hexdigest()
        hash_bin = bin(int(h, 16))[2:].zfill(256)  # Convert to binary string

        # Take first final_length bits
        final_bits_str = hash_bin[:final_length]
        final_bits = [int(b) for b in final_bits_str]

        return final_bits


    def measure_qber_direct(self, alice_key):
        """
        For debugging or simulation: measure direct mismatch rate
        """
        m = min(len(self.sifted_key), len(alice_key))
        if m==0:
            return 0.0
        mismatch = sum(self.sifted_key[i] != alice_key[i] for i in range(m))
        return mismatch/m
