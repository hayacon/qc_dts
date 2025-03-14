from user import User
from bell_measurement import Bell_measurement
from post_processing import AlicePostProcessing, BobPostProcessing

import numpy as np

user_1 = User(100)
user_2 = User(100)

basis_1 = user_1.set_basis()
basis_2 = user_2.set_basis()

print('basis_1')
print(basis_1)

state_1, bits_1 = user_1.state_encoder()
state_2, bits_2 = user_2.state_encoder()

#init port processing
alice_pp = AlicePostProcessing(basis_1, bits_1)
bob_pp = BobPostProcessing(basis_2, bits_2)


bsm = Bell_measurement(state_1, state_2)
result = bsm.run_measurements()
# bsm.print_results()
result = bsm.get_interpretation()
print(result)

# post processing
nA = alice_pp.sifting(basis_2, result)
nB = bob_pp.sifting(basis_1, result)

print(f"Sifted length => Alice: {nA}, Bob: {nB}")
print("Alice's sifted key:", alice_pp.sifted_key)
print("Bob's   sifted key:", bob_pp.sifted_key)

key_len = min(len(alice_pp.sifted_key), len(bob_pp.sifted_key))
sample_size = int(0.3*key_len)
reveal_indices = np.random.choice(key_len, sample_size, replace=False)

# Alice reveals only those bits
alice_reveal_data = alice_pp.reveal_subset_bits(reveal_indices)
# Bob compares
qber_est, used_idx = bob_pp.estimate_qber_sample(alice_reveal_data)
print(f"\nQBER estimated on {len(reveal_indices)} bits =>", qber_est)

# Both sides must remove those revealed bits from their final key
alice_pp.remove_indices(reveal_indices)
bob_pp.remove_indices(reveal_indices)

# 5) Perform multi-pass CASCADE-like error correction
    #    e.g. 2 passes, each with block size=4
bob_pp.cascade_ec(alice_obj=alice_pp, pass_block_sizes=[4,4], randomize_passes=True)

print("\nKeys after parity-check EC:")
print("Alice key:", alice_pp.sifted_key)
print("Bob   key:", bob_pp.sifted_key)

#check how of the key matches at this stage, print it in percentage
match_count = sum(a == b for a, b in zip(alice_pp.sifted_key, bob_pp.sifted_key))
match_percent = 100 * match_count / len(alice_pp.sifted_key)
print(f"\nMatching bits: {match_count} / {len(alice_pp.sifted_key)} => {match_percent:.2f}%")

# 6) Final keys
alice_final = alice_pp.privacy_amplification(final_length=16, salt=b"mysharedseed")
bob_final   = bob_pp.privacy_amplification(final_length=16, salt=b"mysharedseed")

print("\nPrivacy-Amplified Keys (16 bits):")
print("Alice final hashed bits:", alice_final)
print("Bob   final hashed bits:", bob_final)
# 7) Check how much of the final key matches
# get percentage of matching bits
match_count = sum(a == b for a, b in zip(alice_final, bob_final))
match_percent = 100 * match_count / len(alice_final)
print(f"\nMatching bits: {match_count} / {len(alice_final)} => {match_percent:.2f}%")

