import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from wcode.net.Vision_Transformer.SAM.model.Sam import Sam


class LoRA_QKV(nn.Module):
    def __init__(
        self,
        QKV: nn.Linear,
        Q_A: nn.Linear,
        Q_B: nn.Linear,
        V_A: nn.Linear,
        V_B: nn.Linear,
    ) -> torch.tensor:
        super().__init__()
        self.QKV = QKV
        self.Q_A = Q_A
        self.Q_B = Q_B
        self.V_A = V_A
        self.V_B = V_B
        self.dim = QKV.in_features
        self.w_identity = torch.eye(QKV.in_features)

    def forward(self, x):
        # B, N, N, 3 * origin_channel
        qkv = self.QKV(x)
        new_q = self.Q_B(self.Q_A(x))
        new_v = self.V_B(self.V_A(x))

        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v

        return qkv


class LoRA_Sam(nn.Module):
    # LoRA_Sam only contain the parameters of LoRA from Sam.image_encoder
    def __init__(self, Sam_model: Sam, r: int) -> None:
        super(LoRA_Sam, self).__init__()

        assert r > 0

        for param in Sam_model.image_encoder.parameters():
            param.requires_grad = False

        self.A_layers = []
        self.B_layers = []
        for idx, blk in enumerate(Sam_model.image_encoder.blocks):
            QKV_layer = blk.attn.qkv
            self.dim = QKV_layer.in_features
            Q_A_layer = nn.Linear(self.dim, r, bias=False)
            self.A_layers.append(Q_A_layer)
            Q_B_layer = nn.Linear(r, self.dim, bias=False)
            self.B_layers.append(Q_B_layer)
            V_A_layer = nn.Linear(self.dim, r, bias=False)
            self.A_layers.append(V_A_layer)
            V_B_layer = nn.Linear(r, self.dim, bias=False)
            self.B_layers.append(V_B_layer)

            blk.attn.qkv = LoRA_QKV(
                QKV_layer, Q_A_layer, Q_B_layer, V_A_layer, V_B_layer
            )

        self.reset_parameters()
        self.sam = Sam_model

    def forward(
        self, batched_input: torch.Tensor, multimask_output: bool, image_size: int
    ):
        return self.sam(batched_input, multimask_output, image_size)

    def reset_parameters(self):
        for A_layer in self.A_layers:
            nn.init.kaiming_uniform_(A_layer.weight, a=math.sqrt(5))
        for B_layer in self.B_layers:
            nn.init.zeros_(B_layer.weight)

    def save_lora_parameters(self, filename: str) -> None:
        """
        Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.A_layers)  # actually, it is half
        a_tensors = {
            f"lora_A_{i:03d}": self.A_layers[i].weight for i in range(num_layer)
        }
        b_tensors = {
            f"lora_B_{i:03d}": self.B_layers[i].weight for i in range(num_layer)
        }

        # save prompt encoder, only 'state_dict', the 'named_parameter' is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        for key, value in state_dict.items():
            if "prompt_encoder" in key:
                prompt_encoder_tensors[key] = value
            if "mask_decoder" in key:
                mask_decoder_tensors[key] = value

        merged_dict = {
            **a_tensors,
            **b_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        """
        Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.A_layers):
            saved_key = f"lora_A_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.B_layers):
            saved_key = f"lora_B_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if "prompt_encoder" in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {
            k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)
        }
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if "mask_decoder" in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {
            k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)
        }
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)
