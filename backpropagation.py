# Forward pass
x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias

# Multiplying inputs by weights
wx0 = x[0] * w[0]
wx1 = x[1] * w[1]
wx2 = x[2] * w[2]

# Adding
s = wx0 + wx1 + wx2 + b

# ReLU
y = max(s, 0)  # we already described that with ReLU activation function description
print('initial loss:', y)

# Backward pass
dy = (1 if s > 0 else 0)  # derivative on ReLU activation function

dwx0 = 1 * dy 
dwx1 = 1 * dy
dwx2 = 1 * dy
db = 1 * dy

dx0 = w[0] * dwx0  
dw0 = x[0] * dwx0
dx1 = w[1] * dwx1
dw1 = x[1] * dwx1
dx2 = w[2] * dwx2
dw2 = x[2] * dwx2

#print(dw0, dw1, dw2, b)

dx = [dx0, dx1, dx2] # gradients on inputs
dw = [dw0, dw1, dw2] # gradients on weights

# Apply gradient to weights
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print('new weights and bias:', w, b)

# Multiplying inputs by weights
wx0 = x[0] * w[0]
wx1 = x[1] * w[1]
wx2 = x[2] * w[2]

# Adding
s = wx0 + wx1 + wx2 + b

# ReLU
y = max(s, 0) 
print('new loss:', y)
