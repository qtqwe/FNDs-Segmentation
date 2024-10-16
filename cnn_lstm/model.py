from utils import *
from attention_module import se_block, cbam_block, eca_block
from thop import profile
import time

attention_blocks = [se_block, cbam_block, eca_block]


class UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, phi=0):
        super(UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.dropout = nn.Dropout(0.2)
        self.convlstm = ConvLSTM(input_size=(12, 12),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

        self.phi = phi
        if 1 <= phi <= 3:
            self.feat1_att = attention_blocks[self.phi - 1](64)
            self.feat2_att = attention_blocks[self.phi - 1](128)
            self.feat3_att = attention_blocks[self.phi - 1](256)
            self.feat4_att = attention_blocks[self.phi - 1](512)

    def forward(self, x):

        x = torch.unbind(x, dim=0)

        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x5 = self.dropout(x5)
            data.append(x5.unsqueeze(0))

        data = torch.cat(data, dim=0)

        ##lstm 1, 5, 512, 8, 16
        lstm, _ = self.convlstm(data)

        test = lstm[0][-1, :, :, :, :]

        if 1 <= self.phi <= 3:
            x1 = self.feat1_att(x1)

            x2 = self.feat2_att(x2)

            x3 = self.feat3_att(x3)

            x4 = self.feat4_att(x4)

        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_ConvLSTM(n_channels=3, n_classes=2, phi=1).to(device)
    input_data = torch.randn(5, 5, 3, 192, 192).to(device)  # Move input data to the device

    # Use thop.profile to calculate FLOPs and Params
    flops, params = profile(model, inputs=(input_data,), verbose=False)
    print(f"Total Params (M): {params / 1e6:.2f}")
    print(f"Total FLOPs (G): {flops / 1e9:.2f}")

    # Calculate memory consumption
    torch.cuda.empty_cache()  # Clear cache
    with torch.no_grad():  # Disable gradient computation
        model(input_data)  # Forward pass to measure peak memory usage
    memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    torch.cuda.reset_peak_memory_stats(device)  # Reset peak memory usage record

    print(f"Memory Usage (MB): {memory_usage:.2f}")

    # Measure inference time
    start_time = time.time()
    model(input_data)  # Forward pass
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference Time (ms): {inference_time:.2f}")
