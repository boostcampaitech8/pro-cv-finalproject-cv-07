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
    
    
def save_log_return_plot(y_true, y_pred, horizons, save_dir, filename="log_return_plot.png"):
    for h in range(len(horizons)):
        plt.figure(figsize=(8,3))
        plt.plot(np.arange(len(y_true)), y_true[:, h], label="true")
        plt.plot(np.arange(len(y_pred)), y_pred[:, h], label="pred")
        plt.title(f"Log-Return plot - Horizon {horizons[h]} day(s)")
        plt.legend()
        plt.tight_layout()

        fn = f"{horizons[h]}_" + filename
        save_path = os.path.join(save_dir, fn)
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"📈 Log Return plot saved to: {save_path}")
    
    
def save_close_plot(dates, actual_close, pred_close, horizons, save_dir, filename="close_plot.png"):
    for h in range(len(horizons)):
        plt.figure(figsize=(14, 6))
        plt.plot(dates, actual_close[h, :], label="Actual Close")
        plt.plot(dates, pred_close[h, :], label="Predicted Close", alpha=0.8)

        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title(f"Actual vs Predicted Close Price - Horizon {horizons[h]} day(s)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fn = f"{horizons[h]}_" + filename
        save_path = os.path.join(save_dir, fn)
        plt.savefig(save_path, dpi=300)
        plt.close()
    
        print(f"📈 Close plot saved to: {save_path}")