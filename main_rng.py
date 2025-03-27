from rng import RNG

def test_result(chi, 
                entropy,
                auto_corr,
                cumsum_forward,
                cumsum_backward,
                random_excursions,
                random_excursions_variant,
                binary_matrix_rank,
                non_overlapping_template,
                overlapping_template,
                linear_complexity,
                universal):
    """
    Prints test results for Chi-square, Shannon entropy, autocorrelation, and NIST tests.
    Each test result includes its key metric, p-value(s), and 'Is random?' status in blue (True) or red (False).
    Final output indicates if the sequence is truly random based on all tests passing.
    """
    # Original tests
    print("--")
    print(f'Chi square statistics: {chi["chi_square_stat"]}')
    print(f'P-value: {chi["p_value"]}')
    print(f'Is random?: \033[94m{chi["is_random"]}\033[0m' if chi["is_random"] else f'Is random?: \033[31m{chi["is_random"]}\033[0m')
    print('--')
    print(f'Shannon entropy: {entropy["shannon_entropy"]}')
    print(f'Is random?: \033[94m{entropy["is_random"]}\033[0m' if entropy["is_random"] else f'Is random?: \033[31m{entropy["is_random"]}\033[0m')
    print('--')
    print(f'Auto-correlation values: {auto_corr["autocorrelation_values"]}')
    print(f'Is random?: \033[94m{auto_corr["is_random"]}\033[0m' if auto_corr["is_random"] else f'Is random?: \033[31m{auto_corr["is_random"]}\033[0m')

    # NIST tests
    
    print('--')
    print(f'Cumulative Sums (Forward) P-value: {cumsum_forward["p_value"]:.5f}')
    print(f'Is random?: \033[94m{cumsum_forward["is_random"]}\033[0m' if cumsum_forward["is_random"] else f'Is random?: \033[31m{cumsum_forward["is_random"]}\033[0m')
    
    print('--')
    print(f'Cumulative Sums (Backward) P-value: {cumsum_backward["p_value"]:.5f}')
    print(f'Is random?: \033[94m{cumsum_backward["is_random"]}\033[0m' if cumsum_backward["is_random"] else f'Is random?: \033[31m{cumsum_backward["is_random"]}\033[0m')
    
    print('--')
    print(f'Random Excursions P-values: {random_excursions["p_value"]}')
    print(f'Is random?: \033[94m{random_excursions["is_random"]}\033[0m' if random_excursions["is_random"] else f'Is random?: \033[31m{random_excursions["is_random"]}\033[0m')
    
    print('--')
    print(f'Random Excursions Variant P-values: {random_excursions_variant["p_value"]}')
    print(f'Is random?: \033[94m{random_excursions_variant["is_random"]}\033[0m' if random_excursions_variant["is_random"] else f'Is random?: \033[31m{random_excursions_variant["is_random"]}\033[0m')
    
    print('--')
    print(f'Binary Matrix Rank P-value: {binary_matrix_rank["p_value"]}')
    print(f'Is random?: \033[94m{binary_matrix_rank["is_random"]}\033[0m' if binary_matrix_rank["is_random"] else f'Is random?: \033[31m{binary_matrix_rank["is_random"]}\033[0m')
    
    print('--')
    print(f'Non-overlapping Template P-value: {non_overlapping_template["p_value"]}')
    print(f'Is random?: \033[94m{non_overlapping_template["is_random"]}\033[0m' if non_overlapping_template["is_random"] else f'Is random?: \033[31m{non_overlapping_template["is_random"]}\033[0m')
    
    print('--')
    print(f'Overlapping Template P-value: {overlapping_template["p_value"]}')
    print(f'Is random?: \033[94m{overlapping_template["is_random"]}\033[0m' if overlapping_template["is_random"] else f'Is random?: \033[31m{overlapping_template["is_random"]}\033[0m')
    
    print('--')
    print(f'Linear Complexity P-value: {linear_complexity["p_value"]}')
    print(f'Is random?: \033[94m{linear_complexity["is_random"]}\033[0m' if linear_complexity["is_random"] else f'Is random?: \033[31m{linear_complexity["is_random"]}\033[0m')
    
    print('--')
    print(f'Universal Test P-value: {universal["p_value"]}')
    print(f'Is random?: \033[94m{universal["is_random"]}\033[0m' if universal["is_random"] else f'Is random?: \033[31m{universal["is_random"]}\033[0m')
    
    print('=====>')

    # Combine all outcomes
    outcomes = [chi, entropy, auto_corr, cumsum_forward, cumsum_backward,
                random_excursions, random_excursions_variant, binary_matrix_rank,
                non_overlapping_template, overlapping_template, linear_complexity, universal]
    is_random = all(outcome["is_random"] for outcome in outcomes)

    if is_random:
        color = "\033[94m"  # Blue
    else:
        color = "\033[31m"  # Red

    print(f'Is the generated random bit truly random?: {color}{is_random}\033[0m')


# bit_len = 2000
bit_len = 1000

qrng = RNG(bit_len=bit_len, options='qrng')
print('Quantum random number generator')
qrng.generate_random_bits()
chi_result = qrng.chi_square_test()
entropy_result = qrng.shannon_entropy()
auto_corr_result = qrng.auto_correlation()
cumulative_sums_forward_result, cumulative_sums_backward_result = qrng.cumulative_sums()
random_excursions_result = qrng.random_excursions()
random_excursions_variant_result = qrng.random_excursions_variant()
binary_matrix_rank_result = qrng.binary_matrix_rank()
non_overlapping_template_result = qrng.non_overlapping_template_matching()
overlapping_template_result = qrng.overlapping_template_matching()
linear_complexity_result = qrng.linear_complexity()
universal_result = qrng.maurers_universal()
test_result(
    chi = chi_result,
    entropy = entropy_result,
    auto_corr = auto_corr_result,
    cumsum_forward = cumulative_sums_forward_result,
    cumsum_backward = cumulative_sums_backward_result,
    random_excursions = random_excursions_result,
    random_excursions_variant = random_excursions_variant_result,
    binary_matrix_rank = binary_matrix_rank_result,
    non_overlapping_template = non_overlapping_template_result,
    overlapping_template = overlapping_template_result,
    linear_complexity = linear_complexity_result,
    universal = universal_result
)

print('============================================')
numpy_rng = RNG(bit_len=bit_len, options='numpy')
print('Numpy random number generator')
numpy_rng.generate_random_bits()
chi_result = numpy_rng.chi_square_test()
entropy_result = numpy_rng.shannon_entropy()
auto_corr_result = numpy_rng.auto_correlation()
cumulative_sums_forward_result, cumulative_sums_backward_result = numpy_rng.cumulative_sums()
random_excursions_result = numpy_rng.random_excursions()
random_excursions_variant_result = numpy_rng.random_excursions_variant()
binary_matrix_rank_result = numpy_rng.binary_matrix_rank()
non_overlapping_template_result = numpy_rng.non_overlapping_template_matching()
overlapping_template_result = numpy_rng.overlapping_template_matching()
linear_complexity_result = numpy_rng.linear_complexity()
universal_result = numpy_rng.maurers_universal()
test_result(
    chi = chi_result,
    entropy = entropy_result,
    auto_corr = auto_corr_result,
    cumsum_forward = cumulative_sums_forward_result,
    cumsum_backward = cumulative_sums_backward_result,
    random_excursions = random_excursions_result,
    random_excursions_variant = random_excursions_variant_result,
    binary_matrix_rank = binary_matrix_rank_result,
    non_overlapping_template = non_overlapping_template_result,
    overlapping_template = overlapping_template_result,
    linear_complexity = linear_complexity_result,
    universal = universal_result
)


print('============================================')
os_rng = RNG(bit_len=bit_len, options='os')
print('OS random number generator')
os_rng.generate_random_bits()
chi_result = os_rng.chi_square_test()
entropy_result = os_rng.shannon_entropy()
auto_corr_result = os_rng.auto_correlation()
cumulative_sums_forward_result, cumulative_sums_backward_result = os_rng.cumulative_sums()
random_excursions_result = os_rng.random_excursions()
random_excursions_variant_result = os_rng.random_excursions_variant()
binary_matrix_rank_result = os_rng.binary_matrix_rank()
non_overlapping_template_result = os_rng.non_overlapping_template_matching()
overlapping_template_result = os_rng.overlapping_template_matching()
linear_complexity_result = os_rng.linear_complexity()
universal_result = os_rng.maurers_universal()
test_result(
    chi = chi_result,
    entropy = entropy_result,
    auto_corr = auto_corr_result,
    cumsum_forward = cumulative_sums_forward_result,
    cumsum_backward = cumulative_sums_backward_result,
    random_excursions = random_excursions_result,
    random_excursions_variant = random_excursions_variant_result,
    binary_matrix_rank = binary_matrix_rank_result,
    non_overlapping_template = non_overlapping_template_result,
    overlapping_template = overlapping_template_result,
    linear_complexity = linear_complexity_result,
    universal = universal_result
)


print('============================================')
secrets_rng = RNG(bit_len=bit_len, options='secrets')
print('Secrets random number generator')
secrets_rng.generate_random_bits()
chi_result = secrets_rng.chi_square_test()
entropy_result = secrets_rng.shannon_entropy()
auto_corr_result = secrets_rng.auto_correlation()
cumulative_sums_forward_result, cumulative_sums_backward_result = secrets_rng.cumulative_sums()
random_excursions_result = secrets_rng.random_excursions()
random_excursions_variant_result = secrets_rng.random_excursions_variant()
binary_matrix_rank_result = secrets_rng.binary_matrix_rank()
non_overlapping_template_result = secrets_rng.non_overlapping_template_matching()
overlapping_template_result = secrets_rng.overlapping_template_matching()
linear_complexity_result = secrets_rng.linear_complexity()
universal_result = secrets_rng.maurers_universal()
test_result(
    chi = chi_result,
    entropy = entropy_result,
    auto_corr = auto_corr_result,
    cumsum_forward = cumulative_sums_forward_result,
    cumsum_backward = cumulative_sums_backward_result,
    random_excursions = random_excursions_result,
    random_excursions_variant = random_excursions_variant_result,
    binary_matrix_rank = binary_matrix_rank_result,
    non_overlapping_template = non_overlapping_template_result,
    overlapping_template = overlapping_template_result,
    linear_complexity = linear_complexity_result,
    universal = universal_result
)


print('============================================')
random_rng = RNG(bit_len=bit_len, options='random')
print('Random random number generator')
random_rng.generate_random_bits()
chi_result = random_rng.chi_square_test()
entropy_result = random_rng.shannon_entropy()
auto_corr_result = random_rng.auto_correlation()
cumulative_sums_forward_result, cumulative_sums_backward_result = random_rng.cumulative_sums()
random_excursions_result = random_rng.random_excursions()
random_excursions_variant_result = random_rng.random_excursions_variant()
binary_matrix_rank_result = random_rng.binary_matrix_rank()
non_overlapping_template_result = random_rng.non_overlapping_template_matching()
overlapping_template_result = random_rng.overlapping_template_matching()
linear_complexity_result = random_rng.linear_complexity()
universal_result = random_rng.maurers_universal()
test_result(
    chi = chi_result,
    entropy = entropy_result,
    auto_corr = auto_corr_result,
    cumsum_forward = cumulative_sums_forward_result,
    cumsum_backward = cumulative_sums_backward_result,
    random_excursions = random_excursions_result,
    random_excursions_variant = random_excursions_variant_result,
    binary_matrix_rank = binary_matrix_rank_result,
    non_overlapping_template = non_overlapping_template_result,
    overlapping_template = overlapping_template_result,
    linear_complexity = linear_complexity_result,
    universal = universal_result
)

print('============================================')
crypto_rng = RNG(bit_len=bit_len, options='crypto')
print('Crypto random number generator')
crypto_rng.generate_random_bits()
chi_result = crypto_rng.chi_square_test()
entropy_result = crypto_rng.shannon_entropy()
auto_corr_result = crypto_rng.auto_correlation()
cumulative_sums_forward_result, cumulative_sums_backward_result = crypto_rng.cumulative_sums()
random_excursions_result = crypto_rng.random_excursions()
random_excursions_variant_result = crypto_rng.random_excursions_variant()
binary_matrix_rank_result = crypto_rng.binary_matrix_rank()
non_overlapping_template_result = crypto_rng.non_overlapping_template_matching()
overlapping_template_result = crypto_rng.overlapping_template_matching()
linear_complexity_result = crypto_rng.linear_complexity()
universal_result = crypto_rng.maurers_universal()
test_result(
    chi = chi_result,
    entropy = entropy_result,
    auto_corr = auto_corr_result,
    cumsum_forward = cumulative_sums_forward_result,
    cumsum_backward = cumulative_sums_backward_result,
    random_excursions = random_excursions_result,
    random_excursions_variant = random_excursions_variant_result,
    binary_matrix_rank = binary_matrix_rank_result,
    non_overlapping_template = non_overlapping_template_result,
    overlapping_template = overlapping_template_result,
    linear_complexity = linear_complexity_result,
    universal = universal_result
)
