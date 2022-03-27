import torch
from model.train_model import Video_Encoder_Model, Video_Decoder_Model
import time

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=12, pretrained=True)
Decoder_Model = Video_Decoder_Model()

start_time = time.time()

for i in range(5):
    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 3, 256, 256)
    x3 = torch.rand(1, 3, 256, 256)
    x4 = torch.rand(1, 3, 256, 256)
    x_s = [x1, x2, x3, x4]
    block = Encoder_Model(x_s)
    out1, out2, out3, out4, _ = Decoder_Model(block)

end_time = time.time()

print(f"time_cost: {(end_time - start_time) / 5}")
