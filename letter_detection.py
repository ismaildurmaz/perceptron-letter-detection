from tkinter import *
import numpy as np
import math
import tkinter.filedialog as fdialog
import os

# INITIAL DEFINITIONS
w = 7
h = 9
input_size = w * h
rect_size = 50
states = np.zeros((w, h))
weights = {}
bias = {}

root = Tk()

frame = Frame(root, width=800, height=600)
frame.pack_propagate(0) # don't shrink
frame.pack()

toolbar = Frame(frame, height=50)
toolbar.pack_propagate(0)
toolbar.pack(fill=X)

def open_file(file):
    result = np.zeros((w,h))
    lines = [line.rstrip('\n') for line in open(file)]
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            result[x, y] = 1 if ch == '*' else 0
    return result

# LOAD BUTTON
def loadCallback():
    global states
    file = fdialog.askopenfilename()
    if file != '':
        states = open_file(file)
    print_grid()

Button(toolbar, text="Load", command = loadCallback).pack(side=LEFT)

# SAVE BUTTON
def saveCallback():
    file = fdialog.asksaveasfile(mode='w', defaultextension=".txt")

    for y in range(h):
        for x in range(w):
            file.write('.' if states[x,y] == 0 else '*')
        file.write('\n')
    file.close()

Button(toolbar, text="Save", command = saveCallback).pack(side=LEFT)

# CANVAS GRID
def mouseClick(event):
    x = math.floor(event.x / rect_size)
    y = math.floor(event.y / rect_size)
    if x < w and y < h: states[x, y] = 0 if states[x, y] > 0 else 1 # swap zero & one
    print_grid()
canvas = Canvas(frame, width=500, height=500)
canvas.bind("<Button-1>", mouseClick)
canvas.pack(side=LEFT)

result = Text(frame, width=300, height=500)
result.pack(side=LEFT)

# DRAW GRID
def print_grid():
    for i in range(w):
        for j in range(h):
            if states[i, j] > 0:
                color = 'black'
            else:
                color = 'white'
            canvas.create_rectangle(i * rect_size, j * rect_size, (i + 1) * rect_size, (j + 1) * rect_size, outline="black", fill=color)
print_grid();

# ALGORITHM
Label(toolbar, text='Learning Rate').pack(side=LEFT, padx = 10)
learning_rate_field = Entry(toolbar, textvariable=StringVar(root, value='0.001'), width=8)
learning_rate_field.pack(side=LEFT)

Label(toolbar, text='Threshold').pack(side=LEFT, padx = 10)
threshold_field = Entry(toolbar, textvariable=StringVar(root, value='0.5'), width=8)
threshold_field.pack(side=LEFT)

Label(toolbar, text='Max. Iterations').pack(side=LEFT, padx = 10)
max_iterations_field = Entry(toolbar, textvariable=StringVar(root, value='1000'), width=8)
max_iterations_field.pack(side=LEFT)

"""
 A L G O R I T H M
"""

def train_sample(sample, target, input_size, threshold, LR, max_iterations, bias, weights):
    for it in range(max_iterations):
        trained = True
        for letter, t in target.items():    # visit all letters
            y_in = bias[letter]  # get bias value
            for i in range(input_size):
                y_in += sample[i] * weights[letter][i] # matrix value

            if y_in > threshold:
                y = 1
            elif y_in < -threshold:
                y = -1
            else:
                y = 0
            if y != t:
                bias[letter] = bias[letter] + LR * (t - y)
                for i in range(input_size):
                    weights[letter][i] = weights[letter][i] + LR * sample[i] * (t - y)
                trained = False
        if trained: # if trained break the iteration
            break
    return (trained, it)

"""
    input : Input x values
    input_size: x values count
    threshold : Threshold value for activating neuron
    LR : learning rate
    max_iterations: maximum iterations
"""
def train(input, input_size, threshold = 0.5, LR = 0.001, max_iterations = 10000):
    weights = {}   # weights
    bias = {}      # bias

    # set initial values to zero
    for key in input:
        weights[key] = np.zeros(input_size)
        bias[key] = 0

    #train data
    for key, samples in input.items():        # visit all letters
        for sample_index, sample in enumerate(samples): # letter samples

            target = {} # init target
            for tkey in input:
                target[tkey] = 1 if tkey == key else -1

            trained, it = train_sample(sample, target, input_size, threshold, LR, max_iterations, bias, weights)

            if trained == False:
                log = "%s[%d] is not trained !!!" % (key, sample_index)
            else:
                log = "%s[%d] Iteration: %d" % (key, sample_index, it)
            print(log)
            result.insert(END, log + '\n')
    return (weights, bias)


def trainCallback():
    global weights, bias

    result.delete(1.0, END)

    # fetch training data
    dir_path = os.path.dirname(os.path.realpath('__file__')) + '\\data\\'
    data = {}
    for file in os.listdir(dir_path):
        if file.endswith('.txt'):   # only txt files are acceptable
            ch = file[0].upper()
            if not ch in data:
                data[ch] = []
            matrix = open_file(dir_path + file) # load file to matrix
            matrix = matrix.reshape(input_size) # change to single dimension
            matrix[matrix == 0] = -1     # update to bipolar
            data[ch].append(matrix)      # insert train data into dictionary

    weights, bias = train(
            data,
            input_size,
            float(threshold_field.get()),
            float(learning_rate_field.get()),
            int(max_iterations_field.get()))

Button(toolbar, text="Train", command = trainCallback).pack(side=LEFT)

def testCallback():
    print(test_result_field.get())
    threshold = float(threshold_field.get())
    found = []
    input = states.reshape(input_size)
    input[input == 0] = -1  # update to bipolar
    for letter, weight in weights.items():
        y_in = bias[letter]
        for s, w in zip(input, weight):
            y_in += s * w
        if y_in > threshold:
            found.append(letter)
    if len(found) > 0:
        test_result_field_value.set(', '.join(found))
    else:
        test_result_field_value.set('???')



Button(toolbar, text="Test", command = testCallback).pack(side=LEFT)

test_result_field_value = StringVar()
test_result_field = Entry(toolbar, width=10, textvariable=test_result_field_value)
test_result_field.pack(side=LEFT)

root.mainloop()