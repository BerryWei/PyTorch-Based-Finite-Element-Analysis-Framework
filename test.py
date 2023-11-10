L=30
E=3.0e+7
I=0.0833
A=1
rho = 0.00073
v=0.3

# I=1/12*b*h**3

b = 1
h = 0.99986


print(1/12*b*h**3)
print(b/2)

w1 = 3.516/L/L * (E*I/rho/A)**(0.5)
print(w1)