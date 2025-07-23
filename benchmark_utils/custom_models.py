from deepinv.models import UNet

class MRIUNet(UNet):
    def __init__(self, in_channels, out_channels, scales=3, batch_norm=False):
        self.name = "MRIUNet"
        self.in_channels = in_channels

        super().__init__(in_channels=in_channels, out_channels=out_channels, scales=scales, batch_norm=batch_norm)

    def forward(self, x, sigma=None, **kwargs):
        # Reshape for MRI specific processing
        x = x.reshape(1, self.in_channels, x.shape[3], x.shape[4])

        x = super().forward(x, sigma=sigma, **kwargs)

        return x
