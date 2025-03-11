from user import User
from bell_measurement import Bell_measurement

user_1 = User(30)
user_2 = User(30)

basis_1 = user_1.set_basis()
basis_2 = user_2.set_basis()

state_1 = user_1.state_encoder()
state_2 = user_2.state_encoder()

bell = Bell_measurement(state_1, state_2, basis_1, basis_2)
outcome_1, outcome_2 = bell.measurement()
print(outcome_1)
print(outcome_2)
result = bell.annunce_result()
print(result)
