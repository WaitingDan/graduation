import matplotlib.pyplot as plt
import os


def plot_curve(train_list, val_list, title, model_name):

    plt.figure()

    plt.plot(train_list, label="train")

    plt.plot(val_list, label="val")

    plt.xlabel("epoch")

    plt.ylabel(title)

    plt.title(model_name + "_" + title)

    plt.legend()

    # 保存到 outputs/ 目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, model_name + "_" + title + "_curve.png")

    plt.savefig(save_path)

    plt.close()

    print("Saved:", save_path)