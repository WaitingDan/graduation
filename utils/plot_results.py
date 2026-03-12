import matplotlib.pyplot as plt


def plot_curve(train_list, val_list, title, model_name):

    plt.figure()

    plt.plot(train_list, label="train")

    plt.plot(val_list, label="val")

    plt.xlabel("epoch")

    plt.ylabel(title)

    plt.title(model_name + "_" + title)

    plt.legend()

    save_path = model_name + "_" + title + "_curve.png"

    plt.savefig(save_path)

    plt.close()

    print("Saved:", save_path)