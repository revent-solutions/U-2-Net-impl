import cv2
import numpy as np
import torch

from model import U2NET

model_dir = 'u2net.pth'

# load u2net_portrait model
net = U2NET(3,1)
net.load_state_dict( torch.load('u2net.pth', map_location='cpu'))
if torch.cuda.is_available():
    net.cuda()
net.eval()

self = net
import torchvision
import torch.nn.functional as F

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

def forward(x):
    # print(x.shape)
    ori_shape = x.shape[2:]
    x = torchvision.transforms.Resize((224,224))(x)
    # x = x.float()
    # x = x/255
    # x[:,2,:,:] = (x[:,2,:,:]-0.406)/0.225
    # x[:,1,:,:] = (x[:,1,:,:]-0.456)/0.224
    # x[:,0,:,:] = (x[:,0,:,:]-0.485)/0.229

    
    hx = x

    #stage 1
    hx1 = self.stage1(hx)
    hx = self.pool12(hx1)

    #stage 2
    hx2 = self.stage2(hx)
    hx = self.pool23(hx2)

    #stage 3
    hx3 = self.stage3(hx)
    hx = self.pool34(hx3)

    #stage 4
    hx4 = self.stage4(hx)
    hx = self.pool45(hx4)

    #stage 5
    hx5 = self.stage5(hx)
    hx = self.pool56(hx5)

    #stage 6
    hx6 = self.stage6(hx)
    hx6up = _upsample_like(hx6,hx5)

    #decoder
    hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


    #side output
    d1 = self.side1(hx1d)

    d2 = self.side2(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = self.side3(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = self.side4(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = self.side5(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = self.side6(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
    
#     # normalization
    pred = 1.0 - F.sigmoid(d1)[:,0,:,:]
    
    ma = torch.max(pred)
    mi = torch.min(pred)

    pred = (pred-mi)/(ma-mi)
    # print(pred)
    # pred[pred>0.5] = 1
    # pred[pred<0.5] = 0
    # print(pred)

    # return dn
    pred = torchvision.transforms.Resize(ori_shape)(pred)


    output = pred.squeeze()*255
    output[output<127] = 0
    output[output>127] = 255
    
    # del d1,d2,d3,d4,d5,d6,d0
    return output

    # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
net.forward = forward


import onnx
opset = 12


f = ('u2net.onnx')

output_names = ['output0']
dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)

torch.onnx.export(
    net.cpu(),  
    torch.zeros(1, 3, 224,224),
    f,
    verbose=False,
    opset_version=opset,
    input_names=['images'],
    output_names=output_names,
)

# Checks
model_onnx = onnx.load(f)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model


onnx.save(model_onnx, f)
