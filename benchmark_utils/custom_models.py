from deepinv.models import UNet

class CustomUNet(UNet):
    def __init__(self, in_channels, out_channels, scales=3, batch_norm=False, is_mri=False):
        self.name = "CustomUNet"
        self.in_channels = in_channels
        self.is_mri = is_mri
        
        super().__init__(in_channels=in_channels, out_channels=out_channels, scales=scales, batch_norm=batch_norm)

    def forward(self, x, sigma=None, **kwargs):
        if self.is_mri:
            # Reshape for MRI specific processing
            x = x.reshape(1, self.in_channels, x.shape[3], x.shape[4])
            
        x = super().forward(x, sigma=sigma, **kwargs)
        
        if self.is_mri:
            # Reshape for when MRI
            x = x.reshape(2, 4, x.shape[2], x.shape[3])

        return x
