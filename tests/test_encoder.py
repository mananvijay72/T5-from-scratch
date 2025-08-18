from encoder.encoder_block import EncoderBlock
from core.tensor import Tensor
import numpy as np

def test_encoder_block():
    x = Tensor(np.random.rand(2, 512), requires_grad=True)
    block = EncoderBlock(d_model=512, num_heads=1, d_ff=2048)
    out = block(x)
    assert out.data.shape == (2, 512)