import os
import numpy as np
import matplotlib.pyplot as plt


def save_loss_curve(train_hist, valid_hist, save_dir, filename="loss_curve.png"):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_hist) + 1)

    best_epoch = np.argmin(valid_hist) + 1
    best_val = valid_hist[best_epoch - 1]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_hist, label="Train Loss")
    plt.plot(epochs, valid_hist, label="Valid Loss")
    plt.axvline(x=best_epoch, linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.scatter(best_epoch, best_val, zorder=5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"📈 Loss curve saved to: {save_path}")
    
    
