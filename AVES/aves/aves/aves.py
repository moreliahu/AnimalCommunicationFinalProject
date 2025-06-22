from typing import Union, Tuple

import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchaudio.models import wav2vec2_model

logger = logging.getLogger("aves")
DEFAULT_DTYPE = torch.float32


def load_config(config_path: Union[str, Path]) -> dict:
    """Load the model config json file

    Arguments
    ---------
    config_path: Union[str, Path]
        Path to the model configuration file

    Returns
    -------
        dict: The model configuration
    """
    with open(config_path, "r") as ff:
        obj = json.load(ff)
    return obj


class AVESTorchaudioWrapper(nn.Module):
    """Wrapper for the AVES feature extractor model

    Arguments
    ---------
    config_path: Union[str, Path]
        Path to the model configuration file
    model_path: Union[str, Path]
        Path to the model weights file
    device: str
        Device to run the model on. Defaults to "cuda".
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        model_path: Union[str, Path] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.config = load_config(str(config_path))

        logger.info("Initializing HuBERT model...")
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        if model_path is not None:
            logger.info(f"Loading AVES model weights from {model_path}")
            self.model.load_state_dict(torch.load(str(model_path), weights_only=True))

        self.device = device

    def _prep_input(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        return inputs.to(self.device).to(DEFAULT_DTYPE)

    def forward(
        self, inputs: torch.Tensor, layers: Union[list[int], int, None] = -1
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """For training, use the forward method to get the output of the model.

        Arguments
        ---------
        inputs: torch.Tensor
            Input audio tensor, should have a shape of (batch_size, num_samples)
        layers: Union[list[int], int, None], optional:
            Layer(s) to extract features from. Defaults to -1 (last layer). If None, returns all layers.

        Returns
        -------
            Union[torch.Tensor, list[torch.Tensor]]: Output tensor(s) from the model
        """
        inputs = self._prep_input(inputs)
        out = self.model.extract_features(inputs)[0]

        if layers is not None and isinstance(layers, int):
            return out[layers]

        if layers and isinstance(layers, list):
            return [out[layer] for layer in layers]

        return out

    @torch.no_grad()
    def extract_features(
        self,
        inputs: torch.Tensor,
        layers: Union[list[int], int, None] = -1,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """For inference, use this extract_features method to get the output of the model.

        Arguments
        ---------
            inputs (torch.Tensor): Input tensor of shape (batch_size, num_samples)
            layers (Union[list[int], int, None], optional): Layer(s) to extract features from. Defaults to -1 (last layer).

        Returns
        -------
            Union[torch.Tensor, list[torch.Tensor]]: Output tensor
        """
        return self.forward(inputs, layers)


def load_feature_extractor(
    config_path: Union[str, Path],
    model_path: Union[str, Path] = None,
    device: str = "cuda",
    for_inference: bool = True,
) -> AVESTorchaudioWrapper:
    """Load the AVES feature extractor model

    Arguments
    ---------
    config_path: Union[str, Path]
        Path to the model configuration file
    model_path: Union[str, Path]
        Path to the model weights file. Defaults to None, in which case the original HuBERT weights are used.
    device: str, optional
        Device to run the model on. Defaults to "cuda".
    for_inference: bool, optional
        Whether to set the underlying feature extractor to inference mode. Defaults to True.

    Returns
    -------
        AVESTorchaudioWrapper: The AVES feature extractor model
    """
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    if for_inference:
        return AVESTorchaudioWrapper(config_path, model_path, device).to(device).eval()

    return AVESTorchaudioWrapper(config_path, model_path, device).to(device)


class AVESClassifier(nn.Module):
    """A classifier model using AVES as a feature extractor

    Arguments
    ---------
    config_path: Union[str, Path]
        Path to the model configuration file
    model_path: Union[str, Path]
        Path to the model weights file
    num_classes: int
        Number of target classes
    freeze_feature_extractor: bool, optional
        Whether to freeze the feature extractor. Defaults to True.
    for_inference: bool, optional
        Whether to set the underlying feature extractor to inference mode. Defaults to False.
    device: str, optional
        Device to run the model on. Defaults to "cuda".
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        num_classes: int,
        model_path: Union[str, Path] = None,
        freeze_feature_extractor: bool = True,
        for_inference: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.model = load_feature_extractor(
            config_path, model_path, for_inference=for_inference, device=device
        )
        embeddings_dim = self.model.config.get("encoder_embed_dim", 768)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.device = device
        self.head.to(device)

        if freeze_feature_extractor:
            logger.info(
                "Freezing feature extractor, it will NOT be updated during training!"
            )
            self.model.requires_grad_(False)

        if num_classes == 1:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, torch.Tensor]]:
        out = self.model.forward(inputs, layers=-1)
        out = out.mean(dim=1)  # mean pooling over time dimension
        logits = self.head(out)

        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)

        return loss, logits
