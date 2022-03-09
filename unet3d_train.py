#
from vit_pytorch import *
import torch
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--arg1',dest='arg1', metavar='N', type=int, nargs='+',
                    help='task id')
parser.add_argument('--arg2',dest='arg2', metavar='N', type=int, nargs='+',
                    help='task id')

args = parser.parse_args()
if args.arg1 != None:
    SLURM_ARRAY_TASK_ID = args.arg1[0]
else:
    SLURM_ARRAY_TASK_ID = 1
if args.arg2 != None:
    order = args.arg2[0]
else:
    order = 1
seed = np.random.randint(4e6) + SLURM_ARRAY_TASK_ID + order
np.random.seed(seed)

####################################################
from scipy.special import iv as besseli
import matplotlib.pyplot as plt
import icosahedron as ico#local file
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

import torch
import numpy as np
import pylab as plt
import math

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from vit_pytorch import ViT

def getConvMatricies():
    x3dind = []
    y3dind = []
    z3dind = []

    x2dind = []
    y2dind = []

    for i in range(16):
        for j in range(16):
            for k in range(16):
                x3dind.append(i)
                y3dind.append(j)
                z3dind.append(k)
                
                x2dind.append(i*4+k%4)
                y2dind.append(j*4+k//4)

    return x3dind,y3dind,z3dind,x2dind,y2dind

x3dind,y3dind,z3dind,x2dind,y2dind = getConvMatricies()


#TODO: call from training  get2dfrom3d         

def vmf(mu, kappa, x):
    # single point function
    d = mu.shape[0]
    # compute in the log space
    logvmf = (d//2-1) * np.log(kappa) - np.log((2*np.pi)**(d/2)*besseli(d//2-1,kappa)) + kappa * np.dot(mu,x)
    return np.exp(logvmf)

def apply_vmf(x, mu, kappa, norm=1.0):
    delta = 1.0+vmf(mu, kappa, x)
    y = x * np.vstack([np.power(delta,3)]*x.shape[0])
    return y


def dedup(mat):
    mat = mat - np.mean(mat, axis=1).reshape(3, 1)
    # datetime object containing current date and time
    now = datetime.now()
    start = now.strftime("%d/%m/%Y %H:%M:%S")
    mat = (mat * (1.0 / np.linalg.norm(mat, axis=0).reshape(1, -1)))
    similarities = cosine_similarity(mat.T)
    similarities = similarities >.99999
    similarities = np.triu(similarities)  # upper triangular
    np.fill_diagonal(similarities, 0)  # fill diagonal
    similarities = np.sum(similarities, axis=0)
    similarities = similarities == 0  # keep values that are no one's duplicates
    mat = mat[:, similarities]
    now = datetime.now()
    end = now.strftime("%d/%m/%Y %H:%M:%S")
    return mat

global firstDone
firstDone = False
global baseline
baseline = ico.icosphere(30, 1.3)
ind = (baseline[0, :] ** 2 + baseline[1, :] ** 2 + baseline[2, :] ** 2) >= (
            np.median((baseline[0, :] ** 2 + baseline[1, :] ** 2 + baseline[2, :] ** 2)) - .0001)  # remove zero points
baseline = baseline[:, ind]  # fix to having zero values
baseline = dedup(baseline)


def createOneMutatedIcosphere():
    global firstDone
    global baseline
    numbumps = 50
    w = np.random.rand(numbumps)
    w = w/np.sum(w)


    #xnormed = x/np.linalg.norm(x, axis=0)
    xnormed = baseline#norming in dedup now
    xx = np.zeros_like(xnormed)

    for i in range(numbumps):
        kappa = np.random.randint(1, 200)
        mu = np.random.randn(3); mu = mu/np.linalg.norm(mu)
        y = apply_vmf(xnormed, mu, kappa)
        xx += w[i]*y

    return xx




global numpoints
numpoints = 9002
side = 16
sf = .99999
xs = np.zeros((side,side,side))
ys = np.zeros((side,side,side))
zs = np.zeros((side,side,side))

for i in range(side):
    xs[i,:,:] = i+.5
    ys[:,i,:] = i+.5
    zs[:,:,i] = i+.5

def rasterToXYZ(r):#may need to be between 0 and 7 instead of 0 and side*sf
    #may be better to just keep it between 0 and 1 
    a = np.copy(r)
    xr = (xs * a)[r == 1]
    yr = (ys * a)[r == 1]
    zr = (zs * a)[r == 1]

    #xr = side*sf*(xr - np.min(xr)) * (1.0 / (np.max(xr) - np.min(xr)))
    #yr = side*sf*(yr - np.min(yr)) * (1.0 / (np.max(yr) - np.min(yr)))
    #zr = side*sf*(zr - np.min(zr)) * (1.0 / (np.max(zr) - np.min(zr)))

    #xr = side*xr
    #yr = side*yr
    #zr = side*zr

    return xr,yr,zr

def mutated_icosphere_matrix(length=10,canvas_dim=8):
    points = torch.zeros(length, numpoints, 3).type(torch.FloatTensor)
    canvas = torch.zeros(length,canvas_dim,canvas_dim,canvas_dim).type(torch.FloatTensor)


    for l in range(length):
        if l%100 == 0:
            print(seed,'l',l)
        xx = createOneMutatedIcosphere()
        xx = (xx - np.expand_dims(np.min(xx, axis=1), axis=1)) * np.expand_dims(
            1.0 / (np.max(xx, axis=1) - np.min(xx, axis=1)), axis=1)
        xx = torch.from_numpy(xx)
        xx = xx*sf
        x = xx[0,:]
        y = xx[1,:]
        z = xx[2,:]

        points[l, :, 0] = x[:]  # modified for lstm discriminator
        points[l, :, 1] = y[:]  # modified for lstm discriminator
        points[l, :, 2] = z[:]  # modified for lstm discriminator
        
        canvas[l, (x*side*sf).type(torch.LongTensor), (y*side*sf).type(torch.LongTensor), (z*side*sf).type(torch.LongTensor)] = 1.0

    return {
        'canvas': canvas,
        'points': points.type(torch.FloatTensor)}

def plot_one(fig,img, xx, i=0):
    predres = numpoints
    s = [.001 for x in range(predres)]
    assert len(s) == predres
    c = ['red' for x in range(predres)]
    s = [.01 for x in range(predres)]
    assert len(s) == 9002
    assert len(c) == predres
    ax = fig.add_subplot(10, 10, i + 1,projection='3d')
    ax.set_axis_off()

    redx = xx[:, 0]*side*sf
    redy = xx[:, 1]*side*sf
    redz = xx[:, 2]*side*sf
    ax.scatter(xx[:, 0]*side*sf, xx[:, 1]*side*sf,xx[:, 2]*side*sf, marker=',',  c='red',s=.005,lw=.005)
    gtx,gty,gtz = rasterToXYZ(img)
    ax.scatter(gtx, gty, gtz, marker = ',', c='black',s=.005,lw=.005)



def plot_all(sample=None, model=None, labels=None, i=0):
    if model != None:
        with torch.no_grad():
            global numpoints

            loss, out = mse_vit(sample.cuda(), labels.cuda(), model=model, ret_out=True)
            fig = plt.figure()
            for i in range(mini_batch):
                img = sample[i,0, :, :,:].squeeze().cpu().numpy()
                #X = out[i, :, 0]
                #Y = out[i, :, 1]
                #Z = out[i, :, 2]
                xx = out[i,:,:].cpu().numpy()
                plot_one(fig,img, xx, i=i)
    else:
        fig = plt.figure()
        for i in range(mini_batch):
            img = sample[i,0,:, :,:].squeeze().cpu().numpy()
            xx = labels[i, :,:]
            plot_one(fig,img, xx, i=i)


class MutatedIcospheresDataset(torch.utils.data.Dataset):
    def __init__(self, length=10,canvas_dim = 8):
        canvas_dim=side
        self.length = length
        self.values = mutated_icosphere_matrix(length,canvas_dim)
        self.canvas_dim = canvas_dim
        assert self.values['canvas'].shape[0] == self.length
        assert self.values['points'].shape[0] == self.length

        count = 0
        for i in range(self.length):
            a = self[i]
            c = a[0][0, :, :]
            for el in a[1]:
                y, x = (int)(el[1]), (int)(el[0])

                if x < side - 2 and x > 2 and y < side - 2 and y > 2:
                    if c[y, x] != 1 and \
                            c[y + 1, x] != 1 and c[y + 1, -1 + x] != 1 and c[y + 1, 1 + x] != 1 and \
                            c[y - 1, x] != 1 and c[y, -1 + x] != 1 and c[y, 1 + x] != 1:
                        count += 1
        assert count == 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas = self.values["canvas"]
        canvas = canvas[idx, :, :]
        #canvas = canvas.unsqueeze(1).repeat(1,3,1,1)
        points = self.values["points"]
        points = points[idx, :]
        canvas = canvas.unsqueeze(0)#may not work for inputs of size 1. 
        return canvas, points

    @staticmethod
    def displayCanvas(title, loader, model):
        for sample, labels in loader:
            plot_all(sample=sample, model=model, labels=labels)
            break
        plt.savefig(title, dpi=1200)
        plt.clf()


dataset = MutatedIcospheresDataset(length=20)

mini_batch = 20
loader_demo = data.DataLoader(
    dataset,
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset),
    num_workers=2)
MutatedIcospheresDataset.displayCanvas('mutatedicospheres.png', loader_demo, model=None)




#mini_batch = 20
train_dataset = MutatedIcospheresDataset(length = mini_batch*10)
loader_train = data.DataLoader(
    train_dataset, 
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=train_dataset),
    num_workers=4)

def mse_vit(input, target,model=None,ret_out = False):
    try:
        assert input.shape == (20,1,16,16,16)
        out = model(input)
    except:
        print(seed,'input',input.shape)
        import traceback
        print(seed,traceback.format_exc())
        exit()
    
    out = out.reshape(target.shape)#64, 1000, 2
    assert torch.max(out)<1.1
    assert torch.max(target)<1.1
    
    #out = out#fix this
    if not ret_out:
        return torch.mean((out-target)**2)
    else:
        return torch.mean((out-target)**2),out


dataset = MutatedIcospheresDataset(length = 20)
loader_test = data.DataLoader(
    dataset, 
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset),
    num_workers=4)

dim = np.random.choice([4096])
mlp_dim = np.random.choice([4096])#search over mlp_dim

heads = np.random.choice([i for i in range (14,22)])
depth = np.random.choice([i for i in range (9,13)])#search over depth


try:
    #dim,mlp_dim,heads,depth = 2048,2048,16,6
    model = ViT3d(
        image_size = 16,
        patch_size = 4,
        num_classes = 9002*3,
        dim = dim, 
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels=1
    )
    #img = torch.randn(100,1, 16, 16,16)keep
    #preds = v(img)# (1, 1000)

    model = torch.nn.Sequential(
        model,
        torch.nn.Sigmoid()
    )
    model = model.cuda()
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001, betas = (.9,.999))#ideal
    for epoch in range(10):
        for x,y in loader_train:
            optimizer.zero_grad()
            x = x.cuda()
            assert x.shape == (20,1,16,16,16)

            y = y.cuda()
            loss = mse_vit(x,y,model=model)
            loss.backward()
            optimizer.step()
        print(seed,'epoch',epoch,'loss',loss)

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001, betas = (.9,.999))#ideal
    print('seed is changing learning rate to .0001', seed)
    #for epoch in range(500):
    for epoch in range(40):
        for x,y in loader_train:
            optimizer.zero_grad()
            x = x.cuda()
            #x = get2dfrom3d(x).unsqueeze(1).repeat(1,3,1,1)
            y = y.cuda()
            loss = mse_vit(x,y,model=model)
            loss.backward()
            optimizer.step()
        print(seed,'epoch',epoch,'loss',loss)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001, betas = (.9,.999))#ideal
    print('seed is changing learning rate to .00001', seed)
    #for epoch in range(500):
    for epoch in range(40):
        for x,y in loader_train:
            optimizer.zero_grad()
            x = x.cuda()
            #x = get2dfrom3d(x).unsqueeze(1).repeat(1,3,1,1)
            y = y.cuda()
            loss = mse_vit(x,y,model=model)
            loss.backward()
            optimizer.step()
        print(seed,'epoch',epoch,'loss',loss)

    loss_train = loss.item()

    model = model.eval()
    MutatedIcospheresDataset.displayCanvas('vit-training-3d_{:10.8f}_{}_{}_{}_{}_{}.png'.format(
        loss_train,dim,mlp_dim,heads,depth,seed),loader_train, model = model)

    for x,y in loader_test:
        x = x.cuda()
        assert x.shape == (20,1,16,16,16)
        y = y.cuda()
        loss = mse_vit(x,y,model=model)
        print(seed,'validation loss',loss)
        break

    loss_test = loss.item()
    torch.save(model.state_dict(), '/home/users/washbee1/projects/3d-synthd/models/model_{:10.8f}_{:10.8f}_{}_{}_{}_{}_{}_1000.pth'.format(
        loss_train,loss_test,dim,mlp_dim,heads,depth,seed))

    MutatedIcospheresDataset.displayCanvas('vit-test-set-3d_{:10.8f}_{:10.8f}_{}_{}_{}_{}_{}.png'.format(
        loss_train,loss_test,dim,mlp_dim,heads,depth,seed),loader_test, model = model)

except RuntimeError as e:
    model = None
    torch.cuda.empty_cache()
    print(seed,heads,depth,dim,mlp_dim,'has error')
#https://trendscenter.github.io/wiki/docs/Example_SLURM_scripts#job-array-with-multiple-tasks-on-each-gpu