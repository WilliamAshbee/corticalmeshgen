from unet3dencoder import *
model = UNetEncoder().cuda()
a = torch.randn(32,1,32,32,32).cuda()
print(model(a).shape)