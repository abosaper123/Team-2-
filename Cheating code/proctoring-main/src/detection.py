import time
import matplotlib.pyplot as plt
import numpy as np
import head_pose

PLOT_LENGTH = 200

# Placeholders
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
XDATA = list(range(200))
YDATA = [0] * 200

def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current

def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT, CHEAT_THRESH
    if GLOBAL_CHEAT == 0:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0.1, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.15, PERCENTAGE_CHEAT)
    else:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.5, PERCENTAGE_CHEAT)

    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        print("CHEATING")
    else:
        GLOBAL_CHEAT = 0
    print("Cheat percent: ", PERCENTAGE_CHEAT, GLOBAL_CHEAT)

def run_detection():
    global XDATA, YDATA
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1)
    line, = ax.plot(XDATA, YDATA, 'r-')
    plt.title("Suspicious Behaviour Detection")
    plt.xlabel("Time")
    plt.ylabel("Cheat Probability")

    while True:
        YDATA.pop(0)
        YDATA.append(PERCENTAGE_CHEAT)
        line.set_xdata(XDATA)
        line.set_ydata(YDATA)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.05)  # Short pause to update the plot
        time.sleep(1 / 5)  # Check every 5 seconds
        process()

if __name__ == "__main__":
    run_detection()
