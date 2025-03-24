from user import User, Eve
from bell_measurement import Bell_measurement
from post_processing import AlicePostProcessing, BobPostProcessing

import numpy as np

def main():
    alice = User(100)
    bob = User(100)
    eve = Eve(0.3)

    alice_basis = alice.set_basis()
    bob_basis = bob.set_basis()

    print('Alice: initial basis')
    print(alice_basis)
    print("Bob: initial basis")
    print(bob_basis)
    print("============================================")
    alice_state, alice_bits = alice.state_encoder()
    bob_state, bob_bits = bob.state_encoder()

    print('Alice: initial states')
    print(alice_state)
    print("Bob: initial states")
    print(bob_state)
    print("============================================")

    # #pick who to intercept (alice or bob)
    # intercepted = np.random.choice([0, 1])
    # if intercepted == 0:
    #     intercept_count = eve.intercept(bases=alice_basis, bits=alice_bits)
    # else:
    #     intercept_count = eve.intercept(bases=bob_basis, bits=bob_bits)

    # print(f"Eve intercepted {intercept_count} bits")
    # print("============================================")

    #init port processing
    alice_pp = AlicePostProcessing(alice_basis, alice_bits)
    bob_pp = BobPostProcessing(bob_basis, bob_bits)


    bsm = Bell_measurement(alice_state, bob_state)
    result = bsm.run_measurements()
    # bsm.print_results()
    result = bsm.get_interpretation()
    print("Bell measurement results:")
    print(result)
    print("============================================")

    # post processing
    nA = alice_pp.sifting(bob_basis, result)
    nB = bob_pp.sifting(alice_basis, result)

    print(f"Sifted length => Alice: {nA}, Bob: {nB}")
    print("Alice's sifted key:", alice_pp.sifted_key)
    print("Bob's   sifted key:", bob_pp.sifted_key)
    print("============================================")

    key_len = min(len(alice_pp.sifted_key), len(bob_pp.sifted_key))
    sample_size = int(0.3*key_len)
    reveal_indices = np.random.choice(key_len, sample_size, replace=False)

    # Alice reveals only those bits
    alice_reveal_data = alice_pp.reveal_subset_bits(reveal_indices)
    # Bob compares
    qber_est, used_idx = bob_pp.estimate_qber_sample(alice_reveal_data)
    print(f"\nQBER estimated on {len(reveal_indices)} bits =>", qber_est)
    print("============================================")
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
    print("============================================")
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
    return match_percent

scount = 0
fcount = 0
for i in range(100):
    result = main()
    if result == 100:
        scount += 1
    else:
        fcount += 1

print(f"Success: {scount}, Fail: {fcount}")
success_percentage = (scount / (scount + fcount)) * 100
print(f"Success Percentage: {success_percentage}%")

