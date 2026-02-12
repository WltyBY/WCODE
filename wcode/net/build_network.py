from wcode.net.CNN.VNet.VNet import VNet
from wcode.net.CNN.UNet.UNet import UNet
from wcode.net.CNN.ResUNet.ResUNet import ResUNet
from wcode.net.CNN.DFUNet.DFUNet import DFUNet


def build_network(network_settings: dict):
    if network_settings["label"].lower() == "vnet":
        network_class = VNet
    elif network_settings["label"].lower() == "unet":
        network_class = UNet
    elif network_settings["label"].lower() == "resunet":
        network_class = ResUNet
    elif network_settings["label"].lower() == "dfunet":
        network_class = DFUNet
    else:
        raise ValueError("Unsupport model: {} in official implementations.".format(network_settings["label"]))
    
    return network_class(network_settings)