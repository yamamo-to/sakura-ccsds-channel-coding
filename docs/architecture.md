# Software Architecture & Module Interface

## Module Overview
- `src/ccsds/rs.py`: RS Encoder & Decoder (Berlekamp-Massey or Welch-Berlekamp algorithm)
- `src/ccsds/conv.py`: Convolutional Encoder & Viterbi Decoder (Hard/Soft-decision)
- `src/ccsds/turbo.py`: Turbo Encoder & BCJR (MAP) Decoder
- `src/ccsds/pipeline.py`: Concatenated Encoder/Decoder Pipeline (RS + CONV)

## Common Interface Standard
All encoders and decoders must inherit from abstract base classes in `src/ccsds/base.py`:

```python
class BaseCodec(ABC):
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes binary stream (0s and 1s) or byte stream."""
        pass

    @abstractmethod
    def decode(self, rx_symbols: np.ndarray) -> np.ndarray:
        """Decodes received symbols (supports float LLR for soft-decision)."""
        pass
