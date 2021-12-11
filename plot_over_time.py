
import matplotlib.pyplot as plt
import numpy as np

def plot_over_time(result, net):
    print("Plotting loss over epochs...")
    # Plot loss over epochs
    N = len(result.history['loss'])

    #Plot the model evaluation history
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(20,8))

    #plot loss
    fig.add_subplot(1,2,1)
    plt.title("Training Loss")
    plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
    #plt.plot(np.arange(0, N), result.history["val_mean_iou"], label="val_mean_iou")
    plt.ylim(0, max(result.history["loss"]))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    # plot accuracy
    fig.add_subplot(1,2,2)
    plt.title("Training Accuracy")
    #plt.plot(np.arange(0, N), result.history["categorical_accuracy"], label="train_accuracy")
    #plt.plot(np.arange(0, N), result.history["val_categorical_accuracy"], label="val_accuracy")
    plt.ylim(0, 1)

    # plot mean iou
    fig.add_subplot(1,2,2)
    plt.title("Training mIOU")
    plt.plot(np.arange(0, N), result.history["mean_iou"], label="train_mean_iou")
    plt.plot(np.arange(0, N), result.history["val_mean_iou"], label="val_mean_iou")
    plt.ylim(0, 1)

    plt.xlabel("Epochs")
    plt.ylabel("mIOU")
    plt.legend(loc="lower left")

    # Save plot
    plt.savefig("plots/training_plot_" + net + ".pdf")
    plt.show()
