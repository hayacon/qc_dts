from user import User, Eve
from bell_measurement import Bell_measurement
from post_processing import AlicePostProcessing, BobPostProcessing

import numpy as np
import time
import os
import csv
from tqdm import tqdm

def write_test_result(file_name: str,result_dict: dict):
    file_exists = os.path.isfile(file_name)

    with open(file_name, mode='a') as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        if not file_exists or os.path.getsize(file_name) == 0:
            writer.writeheader()

        writer.writerow(result_dict)


def main(initial_bit_length = 1000, eve_ratio = 0.0, sample_ratio = 0.2, channel_noise = 0.0, detection_noise = 0.0):
    alice = User(initial_bit_length, noise=channel_noise)
    bob = User(initial_bit_length, noise=channel_noise)
    eve = Eve(eve_ratio) 

    alice_basis = alice.set_basis()
    bob_basis = bob.set_basis()

    # print('Alice: initial basis')
    # print(alice_basis)
    # print("Bob: initial basis")
    # print(bob_basis)
    # print("============================================")
    alice_state, alice_bits = alice.state_encoder()
    bob_state, bob_bits = bob.state_encoder()

    # print('Alice: initial states')
    # print(alice_state)
    # print("Bob: initial states")
    # print(bob_state)
    # print("============================================")

    #pick who to intercept (alice or bob)
    if eve_ratio != 0:
        intercepted = np.random.choice([0, 1])
        if intercepted == 0:
            intercept_count = eve.intercept(bases=alice_basis, bits=alice_bits)
        else:
            intercept_count = eve.intercept(bases=bob_basis, bits=bob_bits)

    # print(f"Eve intercepted {intercept_count} bits")
    # print("============================================")

    #init port processing
    start = time.time()
    alice_pp = AlicePostProcessing(alice_basis, alice_bits)
    bob_pp = BobPostProcessing(bob_basis, bob_bits)


    bsm = Bell_measurement(alice_state, bob_state, noise = detection_noise)
    result = bsm.run_measurements()
    # bsm.print_results()
    result = bsm.get_interpretation()
    # print("Bell measurement results:")
    # print(result)
    bm_success_count = 0
    for i in result:
        if i != 'no_BSM':
            bm_success_count += 1
        else:
            continue

    # print(f"Number of success measurement: {success_count}")

    # post processing
    alice_sifted_key, nA= alice_pp.sifting(bob_basis, result)
    bob_sifted_key, nB = bob_pp.sifting(alice_basis, result)

    # print(f"Sifted length => Alice: {nA}, Bob: {nB}")
    # print("Alice's sifted key:", alice_sifted_key)
    # print("Bob's   sifted key:", bob_sifted_key)
    # print("============================================")

    key_len = min(nA, nB)
    sample_size = int(sample_ratio*key_len)
    reveal_indices = np.random.choice(key_len, sample_size, replace=False)

    # Alice reveals only those bits
    alice_reveal_data = alice_pp.reveal_subset_bits(reveal_indices)
    # Bob compares
    qber_est, used_idx = bob_pp.estimate_qber_sample(alice_reveal_data)
    # print(f"\nQBER estimated on {len(reveal_indices)} bits =>", qber_est)
    # print("============================================")

    # 5) Perform multi-pass CASCADE-like error correction
        #    e.g. 2 passes, each with block size=4
    bob_ec_key = bob_pp.cascade_ec(alice_obj=alice_pp, pass_block_sizes=[4,4], randomize_passes=True)

    # print("\nKeys after parity-check error correction:")
    # print("Alice key:", alice_sifted_key)
    # print("Bob   key:", bob_ec_key)

    # Both sides must remove those revealed bits from their final key
    alice_key = alice_pp.remove_indices(reveal_indices)
    bob_key = bob_pp.remove_indices(reveal_indices)

    #check how of the key matches at this stage, print it in percentage
    match_count = sum(a == b for a, b in zip(alice_key, bob_key))
    match_percent = 100 * match_count / len(alice_key)
    # print(f"\nMatching bits: {match_count} / {len(alice_key)} => {match_percent:.2f}%")
    # print("============================================")
    # 6) Final keys
    alice_final = alice_pp.privacy_amplification(salt=b"mysharedseed", qber=qber_est)
    bob_final   = bob_pp.privacy_amplification(salt=b"mysharedseed")
    end = time.time()
    execution_time = end - start

    if len(alice_final) == 0 or len(bob_final) == 0:
        # print("Not enough security to produce secure key")
        # print(f"Execution Time: {execution_time} seconds")
        return 0, 0, 0, 0
    else:
        # print(f"\nPrivacy-Amplified Keys: {len(alice_final)} bits")
        # print("Alice final hashed bits:", alice_final)
        # print("Bob   final hashed bits:", bob_final)
        # 7) Check how much of the final key matches
        # get percentage of matching bits
        match_count = sum(a == b for a, b in zip(alice_final, bob_final))
        match_percent = 100 * match_count / len(alice_final)
        # print(f"\nMatching bits: {match_count} / {len(alice_final)} => {match_percent:.2f}%")
        # print(f"Execution Time: {execution_time} seconds")
        final_key_length = len(alice_final)
        return match_percent, final_key_length, qber_est, bm_success_count


eve_ratio_list = [0, 0.1, 0.2, 0.3]
sample_ratio_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
channel_noise_list = [0, 0.05, 0.1, 0.2, 0.3]
detection_noise_list = [0, 0.05, 0.1, 0.2, 0.3]


# Test different eve ratio
for eve in tqdm(eve_ratio_list, desc="Eve ratio"):
    for sample in tqdm(sample_ratio_list, desc="Sample ratio"):
        for channel in tqdm(channel_noise_list, desc="Channel noise"):
            for detection in tqdm(detection_noise_list, desc="Detection noise"):
                scount = 0
                fcount = 0
                key_length_list = []
                qber_list = []
                initial_bit_length = 100
                for i in range(5):
                    result, final_key_length, qber, bm_success_count = main(initial_bit_length = initial_bit_length, eve_ratio=eve, sample_ratio=sample, channel_noise=channel, detection_noise=detection)
                    key_length_list.append(final_key_length)
                    qber_list.append(qber)
                    if result == 100:
                        scount += 1
                        outcome = 'Success'
                    elif result == 0:
                        fcount += 1
                        outcome = 'Not enough security, fail'
                    else:
                        fcount += 1
                        outcome = 'Fail'

                    result_dict = {
                        "initial bit length": initial_bit_length,
                        "eve_ratio": eve,
                        "sample_ratio": sample,
                        "channel_noise": channel,
                        "detection_noise": detection,
                        "qber": qber,
                        "outcome": outcome,
                        "final key length": final_key_length,
                        "bm_success_count": bm_success_count
                    }
                    write_test_result("result.csv", result_dict)

                success_percentage = (scount / (scount + fcount)) * 100
                key_length_list = list(map(int, key_length_list))
                key_length_ave = sum(key_length_list) / len(key_length_list)
                qber_ave = sum(qber_list) / len(qber_list)
                key_length_std = np.std(key_length_list)
                qber_std = np.std(qber_list)
                key_length_diff = np.max(key_length_list) - np.min(key_length_list)
                qber_diff = np.max(qber_list) - np.min(qber_list)
                overall_result = {
                    "initial bit length": initial_bit_length,
                    "eve_ratio": eve,
                    "sample_ratio": sample,
                    "channel_noise": channel,
                    "detection_noise": detection,
                    "success percentage": success_percentage,
                    "final key length average": key_length_ave,
                    "final key length std": key_length_std,
                    "final key length diff": key_length_diff,
                    "qber average": qber_ave,
                    "qber std": qber_std,
                    "qber diff": qber_diff,
                }
                write_test_result("result_overall.csv", overall_result)


