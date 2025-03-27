import os
import matplotlib
matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt


class logger(object):
    def __init__(self, verbose: bool = False):
        self.logging = {
            "mean_fg_dice": list(),
            "ema_fg_dice": list(),
            "dice_per_class": list(),
            "train_losses": list(),
            "val_losses": list(),
            "learning_rates": list(),
            "epoch_start_timestamps": list(),
            "epoch_end_timestamps": list(),
        }
        self.verbose = verbose

    def log(self, key, value, epoch: int):
        assert key in self.logging.keys() and isinstance(
            self.logging[key], list
        ), "This function is only intended to log stuff to lists and to have one entry per epoch"

        if self.verbose:
            print(f"logging {key}: {value} for epoch {epoch}")

        if len(self.logging[key]) < (epoch + 1):
            self.logging[key].append(value)
        else:
            assert len(self.logging[key]) == (epoch + 1), (
                "something went horribly wrong. My logging "
                "lists length is off by more than 1"
            )
            print(f"maybe some logging issue!? logging {key} and {value}")
            self.logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == "mean_fg_dice":
            new_ema_pseudo_dice = (
                self.logging["ema_fg_dice"][epoch - 1] * 0.9 + 0.1 * value
                if len(self.logging["ema_fg_dice"]) > 0
                else value
            )
            self.log("ema_fg_dice", new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        epoch = (
            min([len(i) for i in self.logging.values()]) - 1
        )  # lists of epoch 0 have len 1
        sns.set_theme(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(
            x_values,
            self.logging["train_losses"][: epoch + 1],
            color="b",
            ls="-",
            label="loss_tr",
            linewidth=4,
        )
        ax.plot(
            x_values,
            self.logging["val_losses"][: epoch + 1],
            color="r",
            ls="-",
            label="loss_val",
            linewidth=4,
        )
        ax2.plot(
            x_values,
            self.logging["mean_fg_dice"][: epoch + 1],
            color="g",
            ls="dotted",
            label="pseudo dice",
            linewidth=3,
        )
        ax2.plot(
            x_values,
            self.logging["ema_fg_dice"][: epoch + 1],
            color="g",
            ls="-",
            label="pseudo dice (mov. avg.)",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(
            x_values,
            [
                i - j
                for i, j in zip(
                    self.logging["epoch_end_timestamps"][: epoch + 1],
                    self.logging["epoch_start_timestamps"],
                )
            ][: epoch + 1],
            color="b",
            ls="-",
            label="epoch duration",
            linewidth=4,
        )
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(
            x_values,
            self.logging["learning_rates"][: epoch + 1],
            color="b",
            ls="-",
            label="learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(os.path.join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.logging

    def load_checkpoint(self, checkpoint: dict):
        self.logging = checkpoint
