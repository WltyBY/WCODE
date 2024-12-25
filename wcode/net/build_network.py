from wcode.net.CNN.VNet.VNet import VNet
from wcode.net.CNN.UNet.UNet import UNet
from wcode.net.CNN.ResUNet.ResUNet import ResUNet
from wcode.net.CNN.DFUNet.DFUNet import DFUNet
from wcode.net.CNN.STUNet.STUNet import build_STUNet
from wcode.net.Vision_Transformer.SAM.build_sam import sam_model_registry
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights


def build_network(network_settings: dict):
    if network_settings["label"].lower() in ["vnet", "unet", "resunet", "dfunet"]:
        if network_settings["label"].lower() == "vnet":
            network_class = VNet
        elif network_settings["label"].lower() == "unet":
            network_class = UNet
        elif network_settings["label"].lower() == "resunet":
            network_class = ResUNet
        elif network_settings["label"].lower() == "dfunet":
            network_class = DFUNet
        else:
            raise Exception("Unsupported network class.")

        if (
            network_settings.__contains__("weight_path")
            and network_settings["weight_path"] is not None
        ):
            net = network_class(network_settings)
            print("Loading weight from:", network_settings["weight_path"])
            load_pretrained_weights(
                net,
                network_settings["weight_path"],
                verbose=True,
            )
            return net
        else:
            return network_class(network_settings)
    elif network_settings["label"].lower() == "stunet":
        if (
            network_settings.__contains__("weight_path")
            and network_settings["weight_path"] is not None
        ):
            print("Loading weight from:", network_settings["weight_path"])
            return build_STUNet(
                network_settings["in_channels"],
                network_settings["out_channels"],
                network_settings["pool_kernel_size"],
                network_settings["deep_supervision"],
                network_settings["model_registry"],
                network_settings["weight_path"],
            )
        else:
            return build_STUNet(
                network_settings["in_channels"],
                network_settings["out_channels"],
                network_settings["pool_kernel_size"],
                network_settings["deep_supervision"],
                network_settings["model_registry"],
            )
    elif network_settings["label"].lower() == "sam":
        if (
            network_settings.__contains__("weight_path")
            and network_settings["weight_path"] is not None
        ):
            print("Loading weight from:", network_settings["weight_path"])
            return sam_model_registry["model_registry"](
                checkpoint=network_settings["weight_path"]
            )
        else:
            return sam_model_registry["model_registry"]()
