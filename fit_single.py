import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from .axisnetworks import *

from .dataset_3d import *
from matplotlib import pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler



def train_single(in_file, out_file, device):
    dataset = OccupancyDataset(in_file)
    print(dataset.data.shape)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = MultiTriplane(1, input_dim=3, output_dim=1)
    #model.net.load_state_dict(torch.load('models/decoder_500_net_only.pt'))
    print('loading decoder 200 net checkpoint')
    model.net.load_state_dict(torch.load('/home/turbo/Qian/Triplane/decoder_200_net_only.pt', map_location='cpu'))
    model = model.to(device)
    #model.net.load_state_dict(torch.load('/home/turbo/Qian/Triplane/test.pt'))

    model.embeddings.train()
    model.net.eval()
    for param in model.net.parameters():
        param.requires_grad = False

    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=40)

    losses = []

    step = 0
    for epoch in range(40):
        start = time.time()
        loss_total = 0
        for X, Y in dataloader:
            X, Y = X.float().to(device), Y.float().to(device)

            preds = model(0, X)
            loss = nn.BCEWithLogitsLoss()(preds, Y)

            # # # DENSITY REG
            rand_coords = torch.rand_like(X) * 2 - 1
            rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
            d_rand_coords = model(0, rand_coords)
            d_rand_coords_offset = model(0, rand_coords_offset)
            loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 6e-1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # if step%50 == 0: print(loss.item())

            loss_total += loss

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch} \t {loss_total.item():01f} \t learning rate: {before_lr:.6f}  --> {after_lr:.6f}")
        #print(time.time() - start)
        losses.append(loss_total.item())
        

    triplane0 = model.embeddings[0].cpu().detach().numpy()
    triplane1 = model.embeddings[1].cpu().detach().numpy()
    triplane2 = model.embeddings[2].cpu().detach().numpy()

    res = np.concatenate((triplane0, triplane1, triplane2))
    np.save(out_file, res)
    print("Triplane Dims: "+str(res.shape))
   




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    train_single(args.input, args.output)
