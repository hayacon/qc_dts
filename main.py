from user import User
from bell_measurement import Bell_measurement

user_1 = User(30)
user_2 = User(30)

basis_1 = user_1.set_basis()
basis_2 = user_2.set_basis()

print('basis_1')
print(basis_1)

state_1, bits_1 = user_1.state_encoder()
state_2, bits_2 = user_2.state_encoder()

bsm = Bell_measurement(state_1, state_2)
result = bsm.run_measurements()
# bsm.print_results()
result = bsm.get_interpretation()
print(result)
