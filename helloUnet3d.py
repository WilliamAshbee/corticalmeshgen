from unet3d import *
model = UNet().cuda()
a = torch.randn(32,1,32,32,32).cuda()
print(model(a))