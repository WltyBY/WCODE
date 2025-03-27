import os
import matplotlib

matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt


class logger(object):
    def __init__(self, verbose: bool = False):
        self.logging = {
            "Generator_train_losses": list(),
            "Discriminator_train_losses": list(),
            "Generator_val_losses": list(),
            "Discriminator_val_losses": list(),
            "ema_val_losses": list(),
            "D_x_val": list(),
            "D_G_z_val": list(),
            "Generator_learning_rates": list(),
            "Discriminator_learning_rates": list(),
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
        if key == "Generator_val_losses":
            new_ema_loss = (
                self.logging["ema_val_losses"][epoch - 1] * 0.9 + 0.1 * value
                if len(self.logging["ema_val_losses"]) > 0
                else value
            )
            self.log("ema_val_losses", new_ema_loss, epoch)

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
            self.logging["Generator_train_losses"][: epoch + 1],
            color="b",
            ls="-",
            label="loss_tr_g",
            linewidth=6,
            alpha=0.6,
        )
        ax.plot(
            x_values,
            self.logging["Discriminator_train_losses"][: epoch + 1],
            color="b",
            ls="-",
            label="loss_tr_d",
            linewidth=3,
        )

        ax.plot(
            x_values,
            self.logging["Generator_val_losses"][: epoch + 1],
            color="r",
            ls="-",
            label="loss_val_g",
            linewidth=6,
            alpha=0.6,
        )
        ax.plot(
            x_values,
            self.logging["Discriminator_val_losses"][: epoch + 1],
            color="r",
            ls="-",
            label="loss_val_d",
            linewidth=3,
        )

        ax.plot(
            x_values,
            self.logging["ema_val_losses"][: epoch + 1],
            color="g",
            ls="-",
            label="pseudo loss (mov. avg.)",
            linewidth=4,
        )

        ax2.plot(
            x_values,
            self.logging["D_x_val"][: epoch + 1],
            color="darkorchid",
            ls="dotted",
            label="mean D(x)",
            linewidth=4,
        )
        ax2.plot(
            x_values,
            self.logging["D_G_z_val"][: epoch + 1],
            color="violet",
            ls="dotted",
            label="mean D(G(z))",
            linewidth=4,
        )

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("performance of Discriminator")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.4, 1))

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
            self.logging["Generator_learning_rates"][: epoch + 1],
            color="b",
            ls="-",
            label="learning rate of Generator",
            linewidth=8,
            alpha=0.5,
        )
        ax.plot(
            x_values,
            self.logging["Discriminator_learning_rates"][: epoch + 1],
            color="b",
            ls="--",
            label="learning rate of Discriminator",
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
