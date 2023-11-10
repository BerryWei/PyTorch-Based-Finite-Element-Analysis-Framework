
P=100
L=0.1
t=0.1
E=30.0e+6
v=0.3
D= E * t*t*t/(12*(1-v*v))
v_max = 0.0056*P*L*L/D
print(v_max)