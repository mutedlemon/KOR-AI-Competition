
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from .modules import Linear


class DecoderInterface(nn.Module):
    def __init__(self):
        super(DecoderInterface, self).__init__()

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p


class BaseDecoder(DecoderInterface):
    """ ASR Decoder Super Class for KoSpeech model implementation """
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, targets: Tensor, encoder_outputs: Tensor, **kwargs) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.
        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, *args) -> Tensor:
        """
        Decode encoder_outputs.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError


class TransducerDecoder(DecoderInterface):
    """ ASR Transducer Decoder Super Class for KoSpeech model implementation """
    def __init__(self):
        super(TransducerDecoder, self).__init__()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propage a `inputs` (targets) for training.
        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor):
            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        raise NotImplementedError


class DecoderRNNT(TransducerDecoder):
    """
    Decoder of RNN-Transducer
    Args:
        num_classes (int): number of classification
        hidden_state_dim (int, optional): hidden state dimension of decoder (default: 512)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of decoder layers (default: 1)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
        dropout_p (float, optional): dropout probability of decoder
    Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    Returns:
        (Tensor, Tensor):
        * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = 'lstm',
            sos_id: int = 1,
            eos_id: int = 2,
            dropout_p: float = 0.2,
    ):
        super(DecoderRNNT, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.out_proj = Linear(hidden_state_dim, output_dim)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor = None,
            hidden_states: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propage a `inputs` (targets) for training.
        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            (Tensor, Tensor):
            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        embedded = self.embedding(inputs)
        outputs, hidden_states = self.rnn(embedded, hidden_states)
        outputs = self.out_proj(outputs)
        return outputs, hidden_states