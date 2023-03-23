from matplotlib import pyplot as plt
import numpy as np

def plot_image(img, prediction, true_label, class_names):
    plt.imshow(img, cmap = plt.cm.binary)

    if np.argmax(prediction) == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel(f"{class_names[np.argmax(prediction)]} {int(np.max(prediction)*100)}% ({class_names[true_label]})", color=color)

def plot_prediction_bars(prediction, true_label):
    bar_plot = plt.bar(np.arange(0, 10), prediction, color="black")
    plt.ylim([0, 1])

    bar_plot[np.argmax(prediction)].set_color("red")
    bar_plot[true_label].set_color("green")

def visualise_prediction(img, prediction, true_label, class_names):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(img, prediction, true_label, class_names)
    plt.subplot(1, 2, 2)
    plot_prediction_bars(prediction, true_label)
    plt.show()