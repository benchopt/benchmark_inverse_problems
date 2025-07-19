import deepinv as dinv

class CustomMSE(dinv.metric.MSE):
    
    transform = lambda x: x
    
    def forward(self, x_net=None, x=None, *args, **kwargs):
        return super().forward(self.transform(x_net), x, *args, **kwargs)


class CustomPSNR(dinv.metric.PSNR):
    
    transform = lambda x: x
    
    def forward(self, x_net=None, x=None, *args, **kwargs):
        return super().forward(self.transform(x_net), x, *args, **kwargs)
