from user import User
from bell_measurement import Bell_measurement

user_1 = User(30)
user_2 = User(30)

basis_1 = user_1.set_basis()
basis_2 = user_2.set_basis()

state_1 = user_1.state_encoder()
state_2 = user_2.state_encoder()
print(state_1)

