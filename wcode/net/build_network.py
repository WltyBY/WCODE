from wcode.net.CNN.VNet import VNet
from wcode.net.CNN.UNet import UNet
from wcode.net.CNN.ResUNet import ResUNet
from wcode.net.CNN.DFUNet import DFUNet
from wcode.net.VisionTranformer.ViT import ViTCLS, ViTSEG

MODEL_FACTORY = {
    "vnet": VNet,
    "unet": UNet,
    "resunet": ResUNet,
    "dfunet": DFUNet,
    "vitcls": ViTCLS,
    "vitseg": ViTSEG,
}


def build_network(network_settings: dict):
    network_name = network_settings["label"].lower()
    if network_name not in MODEL_FACTORY:
        raise ValueError(
            "Unsupport model: {} in official implementations.".format(network_name)
        )

    network_class = MODEL_FACTORY[network_name]
    return network_class(network_settings)
