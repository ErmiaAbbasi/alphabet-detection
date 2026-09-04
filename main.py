from alphabets import *

import numpy as np
import matplotlib.pyplot as plt

# A-Z alphabet data
letters = [
    a, b, c, d, e, f, g, h, i, j, k, l, m,
    n, o, p, q, r, s, t, u, v, w, x, y, z
]

# Convert each 5x6 alphabet into a 1*30 input
x = np.array([
    np.array(letter).reshape(30)
    for letter in letters
])

# 26 one-hot labels: A=0, B=1, ..., Z=25
Y = np.eye(26)

# activation function
def sigmoid(x):
	return(1/(1 + np.exp(-x)))

# Creating the Feed forward neural network
def f_forward(x, w1, w2):
	# hidden
	z1 = x.dot(w1)    # input from layer 1 
	a1 = sigmoid(z1)  # out put of layer 2 
	z2 = a1.dot(w2)   # input of out layer
	a2 = sigmoid(z2)  # output of out layer
	return(a2)

# initializing the weights randomly
def generate_wt(x, Y):
	li =[]
	for i in range(x * Y):
		li.append(np.random.randn())
	return(np.array(li).reshape(x, Y))
	
# for loss we will be using mean square error(MSE)
def loss(out, Y):
	s =(np.square(out-Y))
	s = np.sum(s)/len(Y)
	return(s)

# Back propagation of error 
def back_prop(x, Y, w1, w2, alpha):
	
	# hidden layer
	z1 = x.dot(w1)
	a1 = sigmoid(z1) 
	z2 = a1.dot(w2)
	a2 = sigmoid(z2)
	
	# error in output layer
	d2 =(a2-Y)
	d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), 
								(np.multiply(a1, 1-a1)))
	# Gradient for w1 and w2
	w1_adj = x.transpose().dot(d1)
	w2_adj = a1.transpose().dot(d2)
	
	# Updating parameters
	w1 = w1-(alpha*(w1_adj))
	w2 = w2-(alpha*(w2_adj))
	
	return(w1, w2)

w1 = generate_wt(30, 5)
w2 = generate_wt(5, 26)


def train(x, Y, w1, w2, alpha=0.01, epoch=10):
    acc = []
    losss = []

    for j in range(epoch):
        l = []

        for i in range(len(x)):
            out = f_forward(
                x[i].reshape(1, 30),
                w1,
                w2
            )

            l.append(
                loss(
                    out,
                    Y[i].reshape(1, 26)
                )
            )

            w1, w2 = back_prop(
                x[i].reshape(1, 30),
                Y[i].reshape(1, 26),
                w1,
                w2,
                alpha
            )

        print(
            "epochs:", j + 1,
            "======== acc:",
            (1 - (sum(l) / len(x))) * 100
        )

        acc.append(
            (1 - (sum(l) / len(x))) * 100
        )

        losss.append(
            sum(l) / len(x)
        )

    return acc, losss, w1, w2

acc, losss, w1, w2 = train(x, Y, w1, w2, 0.1, 500)

import matplotlib.pyplot as plt1

plt.figure(figsize=(10, 4))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(acc)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.title("Training Accuracy")

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(losss)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training Loss")

# Adjust spacing
plt.tight_layout()

plt.show()

# =========================
# TEST THE TRAINED NETWORK
# =========================

# Enter a 5x6 letter matrix here
test_letter = [
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0]
]

# Convert 5x6 matrix into 1x30
test_x = np.array(test_letter).reshape(1, 30)

# Give the matrix to the trained neural network
output = f_forward(test_x, w1, w2)

# Find the output neuron with the highest value
prediction = np.argmax(output)

# Convert number back to a letter
predicted_letter = chr(ord('A') + prediction)

print("Predicted letter:", predicted_letter)