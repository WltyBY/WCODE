import torch
from torch import nn

from wcode.net.CNN.baseblock_CNN import ResidualBlock, ConvBlock
from wcode.net.CNN.pooling import ConvDownPool, ConvUpPool
from wcode.net.activate_function import ACTIVATE_LAYER


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_out_channels,
        pool_out_channels,
        dropout_p,
        num_conv,
        kernel_size,
        down_scale_factor,
        normalization,
        activate,
        need_bias,
    ):
        super(DownBlock, self).__init__()

        self.conv = ResidualBlock(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            dropout_p=dropout_p,
            dim=len(kernel_size),
            num_conv=num_conv,
            kernel_size=kernel_size,
            padding_size=[(p - 1) // 2 for p in kernel_size],
            normalization=normalization,
            activate=activate,
            need_bias=need_bias,
        )
        self.downpool = ConvDownPool(
            in_channels=conv_out_channels,
            out_channels=pool_out_channels,
            dim=len(kernel_size),
            pool_kernel_size=down_scale_factor,
            normalization=normalization,
            activate=activate,
            need_bias=need_bias,
        )

    def forward(self, inputs):
        skip_features = self.conv(inputs)
        outputs = self.downpool(skip_features)
        return skip_features, outputs


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        upsample_out_channels,
        conv_out_channels,
        dropout_p,
        num_conv,
        kernel_size,
        up_scale_factor,
        normalization,
        activate,
        need_bias,
    ):
        super(UpBlock, self).__init__()
        self.uppool = ConvUpPool(
            in_channels=in_channels,
            out_channels=upsample_out_channels,
            dim=len(kernel_size),
            pool_kernel_size=up_scale_factor,
            normalization=normalization,
            activate=activate,
            need_bias=need_bias,
        )

        self.conv = ConvBlock(
            in_channels=upsample_out_channels + conv_out_channels,
            out_channels=conv_out_channels,
            dropout_p=dropout_p,
            dim=len(kernel_size),
            num_conv=num_conv,
            kernel_size=kernel_size,
            padding_size=[(p - 1) // 2 for p in kernel_size],
            normalization=normalization,
            activate=activate,
            need_bias=need_bias,
        )
        Activate_layer = ACTIVATE_LAYER[activate.lower()]
        self.activate_layer = Activate_layer()

    def forward(self, inputs, skip_features):
        up_features = self.uppool(inputs)
        outputs = self.conv(torch.cat([skip_features, up_features], 1))
        return self.activate_layer(outputs)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params["in_channels"]
        features = self.params["features"]
        self.dropout_p = self.params["dropout_p"]
        self.num_conv_per_stage = self.params["num_conv_per_stage"]
        self.kernel_size = self.params["kernel_size"]
        self.pool_kernel_size = self.params["pool_kernel_size"]
        self.normalization = self.params["normalization"]
        self.activate = self.params["activate"]
        self.need_bias = self.params["need_bias"]

        self.Encoder_layers = nn.ModuleList()
        for i in range(len(self.num_conv_per_stage) - 1):
            self.Encoder_layers.append(
                DownBlock(
                    in_channels=self.in_channels if i == 0 else features[i],
                    conv_out_channels=features[i],
                    pool_out_channels=features[i + 1],
                    dropout_p=self.dropout_p[i],
                    num_conv=self.num_conv_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    down_scale_factor=self.pool_kernel_size[i],
                    normalization=self.normalization,
                    activate=self.activate,
                    need_bias=self.need_bias,
                )
            )

        self.bottleneck = ConvBlock(
            in_channels=features[-1],
            out_channels=features[-1],
            dropout_p=self.dropout_p[-1],
            dim=len(self.kernel_size[0]),
            num_conv=self.num_conv_per_stage[-1],
            kernel_size=self.kernel_size[-1],
            padding_size=[(p - 1) // 2 for p in self.kernel_size[-1]],
            normalization=self.normalization,
            activate=self.activate,
            need_bias=self.need_bias,
        )

    def forward(self, inputs):
        encoder_out = []
        for i in range(len(self.Encoder_layers)):
            skip_features, inputs = self.Encoder_layers[i](inputs)
            encoder_out.append(skip_features)
        encoder_out.append(self.bottleneck(inputs))
        return encoder_out


class Decoder(nn.Module):
    def __init__(self, params, output_features=False):
        super(Decoder, self).__init__()
        self.params = params
        features = self.params["features"]
        self.kernel_size = self.params["kernel_size"]
        self.pool_kernel_size = self.params["pool_kernel_size"]
        self.dropout_p = self.params["dropout_p"]
        self.num_conv_per_stage = self.params["num_conv_per_stage"]
        self.normalization = self.params["normalization"]
        self.activate = self.params["activate"]
        self.need_bias = self.params["need_bias"]

        self.output_features = output_features

        self.Decoder_layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.Decoder_layers.append(
                UpBlock(
                    in_channels=features[i],
                    upsample_out_channels=features[i + 1],
                    conv_out_channels=features[i + 1],
                    dropout_p=self.dropout_p[i],
                    num_conv=self.num_conv_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    up_scale_factor=self.pool_kernel_size[i],
                    normalization=self.normalization,
                    activate=self.activate,
                    need_bias=self.need_bias,
                )
            )

    def forward(self, encoder_out):
        # low-resolution to high-resolution
        encoder_out = encoder_out[::-1]
        if self.output_features:
            decoder_out = []
            for i, layer in enumerate(self.Decoder_layers):
                outputs = layer(
                    encoder_out[i] if i == 0 else outputs, encoder_out[i + 1]
                )
                decoder_out.append(outputs)
        else:
            for i, layer in enumerate(self.Decoder_layers):
                decoder_out = layer(
                    encoder_out[i] if i == 0 else decoder_out, encoder_out[i + 1]
                )

        return decoder_out


class ResUNet(nn.Module):
    def __init__(self, params):
        super(ResUNet, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)
        self.encoder = Encoder(self.encoder_params)
        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )

        if len(params["kernel_size"][0]) == 2:
            Conv_layer = nn.Conv2d
        elif len(params["kernel_size"][0]) == 3:
            Conv_layer = nn.Conv3d

        if self.deep_supervision:
            self.prediction_head = nn.ModuleList()
            # we will not do deep supervision on the prediction of bottleneck output feature
            # the prediction_heads are from low to high resolution.
            for i in range(1, len(self.encoder_params["num_conv_per_stage"])):
                self.prediction_head.append(
                    Conv_layer(
                        self.decoder_params["features"][i],
                        params["out_channels"],
                        kernel_size=1,
                        bias=params["need_bias"],
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
                bias=params["need_bias"],
            )

    def forward(self, inputs):
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out)
        if self.deep_supervision:
            outputs = []
            for i in range(len(decoder_out)):
                outputs.append(self.prediction_head[i](decoder_out[i]))
            # we assume that the multi-level prediction ranking ranges from high resolution to low resolution
            if self.need_features:
                net_out = {"feature": encoder_out + decoder_out, "pred": outputs[::-1]}
            else:
                net_out = {"pred": outputs[::-1]}
        else:
            if self.need_features:
                outputs = self.prediction_head(decoder_out[-1])
                net_out = {"feature": encoder_out + decoder_out, "pred": outputs}
            else:
                net_out = {"pred": self.prediction_head(decoder_out)}

        return net_out

    def get_EnDecoder_params(self, params):
        encoder_params = {}
        decoder_params = {}

        encoder_params["in_channels"] = params["in_channels"]
        encoder_params["features"] = params["features"]
        encoder_params["dropout_p"] = params["dropout_p"]
        encoder_params["num_conv_per_stage"] = params["num_conv_per_stage"]
        encoder_params["kernel_size"] = params["kernel_size"]
        encoder_params["pool_kernel_size"] = params["pool_kernel_size"]
        encoder_params["normalization"] = params["normalization"]
        encoder_params["activate"] = params["activate"]
        encoder_params["need_bias"] = params["need_bias"]

        assert (
            len(encoder_params["features"])
            == len(encoder_params["dropout_p"])
            == len(encoder_params["num_conv_per_stage"])
            == len(encoder_params["kernel_size"])
            == (len(encoder_params["pool_kernel_size"]) + 1)
        )

        decoder_params["features"] = params["features"][::-1]
        decoder_params["kernel_size"] = params["kernel_size"][::-1]
        decoder_params["pool_kernel_size"] = params["pool_kernel_size"][::-1]
        decoder_params["dropout_p"] = params["dropout_p"][::-1]
        decoder_params["num_conv_per_stage"] = params["num_conv_per_stage"][::-1]
        decoder_params["normalization"] = params["normalization"]
        decoder_params["activate"] = params["activate"]
        decoder_params["need_bias"] = params["need_bias"]

        return encoder_params, decoder_params


if __name__ == "__main__":
    import time

    from wcode.utils.file_operations import open_yaml

    data = open_yaml("./wcode/net/CNN/ResUNet/ResUNet_test.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("-----ResUNet2d-----")
    resunet2d = ResUNet(data["Network2d"]).to(device)
    begin = time.time()
    with torch.no_grad():
        inputs = torch.rand((16, 1, 256, 256)).to(device)
        outputs = resunet2d(inputs)
    print("Time:", time.time() - begin)
    print("Outputs features:")
    if isinstance(outputs["feature"], (list, tuple)):
        for output in outputs["feature"]:
            print(output.shape)
    else:
        print(outputs["feature"].shape)

    print("Outputs prediction:")
    if isinstance(outputs["pred"], (list, tuple)):
        for output in outputs["pred"]:
            print(output.shape)
    else:
        print(outputs["pred"].shape)
    total = sum(p.numel() for p in resunet2d.parameters())
    print("Total params: %.3fM" % (total / 1e6))

    print("-----ResUNet3d-----")
    resunet3d = ResUNet(data["Network3d"]).to(device)
    begin = time.time()
    with torch.no_grad():
        inputs = torch.rand((2, 1, 16, 256, 256)).to(device)
        outputs = resunet3d(inputs)
    print("Time:", time.time() - begin)
    print("Outputs features:")
    if isinstance(outputs["feature"], (list, tuple)):
        for output in outputs["feature"]:
            print(output.shape)
    else:
        print(outputs["feature"].shape)

    print("Outputs prediction:")
    if isinstance(outputs["pred"], (list, tuple)):
        for output in outputs["pred"]:
            print(output.shape)
    else:
        print(outputs["pred"].shape)
    total = sum(p.numel() for p in resunet3d.parameters())
    print("Total params: %.3fM" % (total / 1e6))
