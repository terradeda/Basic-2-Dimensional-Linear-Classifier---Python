__author__ = 'David Terrade'

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def batch_gradient_descent(theta, step_size, improvement, x1, x2, y):
    # ----------------------------------------------------------
    #                  Gradient Descent algorithm
    # ----------------------------------------------------------

    # --------------------------
    #        User Message
    # --------------------------

    print "---------------Batch Gradient Descent---------------"
    print "Training Set Info:"
    print "                     - # Training Examples:", len(y)
    print "                     - # of Input Parameters: 2 \n"

    print "Calculating Model Parameters..."

    time.sleep(1)            # Delay 1 second for user to read the above message

    # --------------------------
    #    Variable Declaration
    # --------------------------

    start = time.time()      # Start Time

    old_theta = [0, 0, 0]      # Vector Containing the parameters before the last iteration
    j_theta_last = 0         # sum of squared errors of new prediction - actual observations from previous iteration
    j_theta_current = 0
    iterations = 0           # Count the number of number of iterations until convergence identified
    decision_boundary = []   # Vector containing the current decision boundary

    # --------------------------
    #  Setup Plotting/Graphing
    # --------------------------

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Decision Boundary for Batch Gradient Descent Model', fontsize=16)   # subplot 111 title
    plt.axes()

    ax.set_xlabel('x1', fontsize=10, fontweight='bold')
    ax.set_ylabel('x2', fontsize=10, fontweight='bold')

    # Plot points
    ax.plot(x1[1:50], x2[1:50], color='r', marker='o',linestyle='None', label="Negative (0)")
    ax.plot(x1[51:100], x2[51:100], color='b', marker='s', linestyle='None', label="Positive (1)")

    # Plot Decision Boundary
    for i in range(int(min(x1)), int(math.ceil(max(x1)))+1):
        decision_boundary.append(float(-theta[1]/theta[2]*i + (0.5-theta[0])/theta[2]))
    line, = ax.plot(range(int(min(x1)), int(math.ceil(max(x1)))+1), decision_boundary, lw=2, label="Decision Boundary")

    # Plot Legend
    plt.legend(loc=4)

    # Show plot
    plt.show()

    # Debugging Message
    # Print "Min",min(x1),"Max",max(x1),"Max2",int(max(x1)), "range",range(int(min(x1)),int(max(x1)))

    # --------------------------
    #          Main Body
    # --------------------------

    # Update theta parameters until the Sum Squared Error between hypothesis and observations Converge
    while abs(improvement) > 0.001:

        j_theta_current = 0
        theta_grad = [0, 0, 0]

        # .......................................
        #         Update Theta Parameters
        # .......................................

        for i in range(0, row_count-1):

            # Calculate the hypothesis for each point given the current parameters
            h_theta = theta[0] + theta[1]*x1[i] + theta[2]*x2[i]

            # calculate the jTheta Gradient
            theta_grad[0] = theta_grad[0] + (y[i]-h_theta)
            theta_grad[1] = theta_grad[1] + (y[i]-h_theta)
            theta_grad[2] = theta_grad[2] + (y[i]-h_theta)

            # Calculate current and cumulative squared errors
            squared_error = pow(h_theta-y[i], 2)
            j_theta_current += squared_error

            # Output Step Summary - Debugging Purposes
            # print "index:", i
            # print "         Inputs:", x2[i], x1[i]
            # print "         Output:", y[i], "\n"
            # print "   Theta Param.:", theta[2], theta[1], theta[0]
            # print "   Model Output:", h_theta, "\n"
            # print "Gradient Values:", theta_grad[0], theta_grad[1],theta_grad[2]
            # print "  Squared Error:", squared_error
            # print " Current J_theta:", j_theta_current/2, "\n"

        j_theta_current = j_theta_current / 2

        # update old theta variables with new theta variables
        old_theta[0] = theta[0]
        old_theta[1] = theta[1]
        old_theta[2] = theta[2]

        theta[0] += step_size*theta_grad[0]           # theta[0] = parameter for constant
        theta[1] += step_size*theta_grad[1]*x1[i]     # theta[1] = parameter for x1
        theta[2] += step_size*theta_grad[2]*x2[i]     # theta[2] = parameter for x2

        # .........................................
        #         Update Decision Boundary
        # .........................................

        # reset the decision boundary
        decision_boundary = []

        for i in range(int(min(x1)), int(math.ceil(max(x1)))+1):

            temp = float(-theta[1]/theta[2]*i + (0.5-theta[0])/theta[2])

            if temp > max(x1):
                decision_boundary.append(np.nan)
            else:
                decision_boundary.append(float(-theta[1]/theta[2]*i + (0.5-theta[0])/theta[2]))

        line.set_ydata(decision_boundary)
        fig.canvas.draw()

        # .....................................
        #   Check for Improvement/Convergence
        # .....................................

        # time.sleep(0.1)
        if iterations == 0:
            improvement = j_theta_current

        else:
            # Check the improvement of model accuracy from previous iteration
            improvement = j_theta_last - j_theta_current

        iterations += 1

        j_theta_last = j_theta_current

    # End Time
    end = time.time()

    # Training Summary
    print "Convergence Detected!"
    print "             Iteration:", iterations
    print "    Total Elapsed Time:", end-start
    print "          Theta Param.:", theta[2], theta[1], theta[0], "(x2,x1,x0)"
    print "                jTheta:", j_theta_current

    plt.show(block=True)


def stochastic_gradient_descent(theta, step_size, improvement, x1, x2, y):
    # -------------------------------------
    # Stochastic Gradient Descent algorithm
    # -------------------------------------

    # --------------------------
    #    Variable Declaration
    # --------------------------
    current_squared_error = 0.0     # Current squared error
    last_squared_error = 0.0        # Previous squared error (for convergence detection)
    iterations = 0                  # Count the number of number of iterations until convergence identified
    decision_boundary = []          # Vector containing the current decision boundary

    # --------------------------
    #  Setup Plotting/Graphing
    # --------------------------
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot Labels
    plt.title('Decision Boundary for Stochastic Gradient Descent Model', fontsize=16)   # subplot 211 title
    ax.set_xlabel('x1', fontsize=10, fontweight='bold')
    ax.set_ylabel('x2', fontsize=10, fontweight='bold')

    # Plot points
    ax.plot(x1[1:50], x2[1:50], color='r', marker='o',linestyle='None', label="Negative (0)")
    ax.plot(x1[51:100], x2[51:100], color='b', marker='s', linestyle='None', label="Positive (1)")

    # Plot Decision Boundary
    for i in range(int(min(x1)), int(math.ceil(max(x1)))+1):
        decision_boundary.append(float(-theta[1]/theta[2]*i + (0.5-theta[0])/theta[2]))
    line, = ax.plot(range(int(min(x1)), int(math.ceil(max(x1)))+1), decision_boundary, lw=2, label="Decision Boundary")

    # Plot Legend
    plt.legend(loc=4)

    # Show plot
    plt.show()

    # Debugging message
    # print "Min",min(x1),"Max",max(x1),"Max2",int(max(x1)), "range",range(int(min(x1)),int(max(x1)))

    i = 0
    while True :

        # ................................
        #     Update Model Parameters
        # ................................

        # Calculate the model prediction for training point i
        h_theta = theta[0] + theta[1]*x1[i] + theta[2]*x2[i]

        # Update the theta parameters using gradient decent using training point i
        theta[0] += step_size*(y[i]-h_theta)
        theta[1] += step_size*(y[i]-h_theta)*x1[i]
        theta[2] += step_size*(y[i]-h_theta)*x2[i]

        # Recalculate the models predictions using the new theta parameters for training point i
        h_theta = theta[0] + theta[1]*x1[i] + theta[2]*x2[i]

        # ................................
        #     Update Decision Boundary
        # ................................

        # Reset the decision boundary
        decision_boundary = []

        # Calculate the Decision Boundary for new parameters
        for j in range(int(min(x1)), int(math.ceil(max(x1)))+1):
            temp = float(-theta[1]/theta[2]*j + (0.5-theta[0])/theta[2])
            if temp > max(x1):
                decision_boundary.append(np.nan)
            else:
                decision_boundary.append(float(-theta[1]/theta[2]*j + (0.5-theta[0])/theta[2]))

        # ................................
        #        Update Plot/Graph
        # ................................
        line.set_ydata(decision_boundary)
        fig.canvas.draw()

        # ................................
        #    Update Mean Squared Error
        # ................................

        # Calculate Squared Error of new parameters over entire training set.
        for j in range(0, row_count-1):

            # Calculate the hypothesis for each point given the current parameters
            h_theta = theta[0] + theta[1]*x1[i] + theta[2]*x2[i]
            current_squared_error += pow(h_theta-y[i], 2)

        current_squared_error = current_squared_error / 2

        # ...................................
        # Check for convergence and exit loop
        # ...................................

        # Calculate Improvement each iteration
        if iterations == 0:
            improvement = current_squared_error
        else:
            improvement = current_squared_error-last_squared_error

        # Check for Convergence
        if abs(improvement) < .1 and abs(current_squared_error) < 50:
            break

        # ...................................
        #           Output Message
        # ...................................
        print "Iteration:", iterations
        print "                index:", i
        print "               Inputs:", x2[i], x1[i]
        print "               Output:", y[i], "\n"
        print "         Theta Param.:", theta[2], theta[1], theta[0]
        print "         Model Output:", h_theta, "\n"
        print "   Last Squared Error:", last_squared_error
        print "Current Squared Error:", current_squared_error
        print "          Improvement:", improvement, "\n"

        # ...................................
        #            Loop Control
        # ...................................

        # Increment current training sample index
        if i == row_count-2:
            i = 0
        else:
            i += 1

        iterations += 1
        last_squared_error = current_squared_error

    plt.show(block=True)


# -------------------------------------
#       Import Data From CSV File
# -------------------------------------
with open('C:/Users/terradeda/Desktop/Machine Learning/Sample Data 1/Classification Data1.csv') as csvfile:

    # input/output variables
    x1 = []
    x2 = []
    y = []

    fileReader = csv.reader(csvfile, delimiter=' ', quotechar='|')      # create a csvFile Reader

    next(csvfile)           # Skip the Column Headers
    row_count = 1

    # ...................
    #     Import Data
    # ...................
    print "Importing data...",
    for row in fileReader:

        # split the row into fields
        fields = row[0].split(",")

        # Append each new row to the corresponding input/output variables
        x1.append(float(fields[0]))
        x2.append(float(fields[1]))
        y.append(float(fields[2]))

        # Output import summary messages
        print "Row#:", row_count, " Data:  X1=", x1[row_count-1], "X2=", x2[row_count-1],  "Y=", y[row_count-1]

        row_count += 1

print "Import Complete \n"
print " Imported:", row_count - 1, "rows"

# -------------------------------------
#        Batch Gradient Descent
# -------------------------------------

theta = [1, 1, 1]       # Vector containing the parameters for out linear equation
stepSize = 0.001        # Size of Step to take each iteration
improvement = 1.0

batch_gradient_descent(theta, stepSize, improvement, x1, x2, y)

# -------------------------------------
#      Stochastic Gradient Descent
# -------------------------------------

theta = [1, 1, 1]         # Vector containing the parameters for out linear equation
stepSize = 0.001        # Size of Step to take each iteration
improvement = 1.0

stochastic_gradient_descent(theta, stepSize, improvement, x1, x2, y)