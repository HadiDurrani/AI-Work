import numpy as np
import math

init_hidden_weight1 = [5, 5, 5]
init_hidden_weight2 = [5, 5, 5]
init_hidden_weight3 = [5, 5, 5]
init_output_weight = [5, 5, 5, 5]

x_inputs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
x2_inp = [x ** 2 for x in x_inputs]
y_out = [5.5, 2, -0.5, -2, -2.5, -2, -0.5, 2, 5.5, 10, 15.5]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feedforward_hid(inp1, inp2, inp_weights, inp_weights2, inp_weights3):
    y1 = inp1 * inp_weights[0] + inp2 * inp_weights[1] + inp_weights[2]
    y1_sig = sigmoid(y1)
    y2 = inp1 * inp_weights2[0] + inp2 * inp_weights2[1] + inp_weights2[2]
    y2_sig = sigmoid(y2)
    y3 = inp1 * inp_weights3[0] + inp2 * inp_weights3[1] + inp_weights3[2]
    y3_sig = sigmoid(y3)
    return [y1_sig, y2_sig, y3_sig]

def feedforward_out(hid_in,weights):
    y = hid_in[0]*weights[0] + hid_in[1]*weights[1] +hid_in[2]*weights[2] + weights[3]
    y_sig = 1/(1+math.exp(-y))
    return y_sig

def error_func(y,ypred):
    err = 0
    for i in range(len(y)):
        err += (y[i] - ypred[i])**2
    return err/len(y)

alpha = 0.01

def backpropagation(x_inputs, x2_inp, y_out, init_hidden_weight1, init_hidden_weight2, init_hidden_weight3, init_output_weight):
    for i in range(len(x_inputs)):
        hid_out = feedforward_hid(x_inputs[i], x2_inp[i], init_hidden_weight1, init_hidden_weight2, init_hidden_weight3)
        output = feedforward_out(hid_out, init_output_weight)
        error = y_out[i] - output
        
        d_output_weights = [hid_out[j] * error * sigmoid_derivative(output) for j in range(len(hid_out))]
        d_output_weights.append(error * sigmoid_derivative(output))  
        
        for j in range(len(init_output_weight)):
            init_output_weight[j] += alpha * d_output_weights[j]
    
        d_hidden_weights1 = [x_inputs[i] * error * sigmoid_derivative(hid_out[0]) for _ in range(2)]
        d_hidden_weights1.append(error * sigmoid_derivative(hid_out[0]))  # For bias weight
        d_hidden_weights2 = [x_inputs[i] * error * sigmoid_derivative(hid_out[1]) for _ in range(2)]
        d_hidden_weights2.append(error * sigmoid_derivative(hid_out[1]))  # For bias weight
        d_hidden_weights3 = [x_inputs[i] * error * sigmoid_derivative(hid_out[2]) for _ in range(2)]
        d_hidden_weights3.append(error * sigmoid_derivative(hid_out[2]))  # For bias weight
        
        for j in range(len(init_hidden_weight1)):
            init_hidden_weight1[j] += alpha * d_hidden_weights1[j]
            init_hidden_weight2[j] += alpha * d_hidden_weights2[j]
            init_hidden_weight3[j] += alpha * d_hidden_weights3[j]
    
    return init_hidden_weight1, init_hidden_weight2, init_hidden_weight3, init_output_weight

for _ in range(1000):
    init_hidden_weight1, init_hidden_weight2, init_hidden_weight3, init_output_weight = backpropagation(x_inputs, x2_inp, y_out, init_hidden_weight1, init_hidden_weight2, init_hidden_weight3, init_output_weight)

output = []
for i in range(len(x_inputs)):
    hid_out = feedforward_hid(x_inputs[i], x2_inp[i], init_hidden_weight1, init_hidden_weight2, init_hidden_weight3)
    output.append(feedforward_out(hid_out, init_output_weight))
error = error_func(y_out, output)
print("Trained Output:", output)
print("Final Error:", error)