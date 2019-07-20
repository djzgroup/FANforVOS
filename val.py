import torch
import argparse, threading, time
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import Image
from train import *
from skimage.morphology import binary_dilation,disk
from crf import densecrf
# from IPython import embed
# from contour_accuracy import *
# import torch.functional as F

palette = Image.open('/mnt/gmy/DAVIS2017/DAVIS-2017-trainval-480p/Annotations/480p/bear/00000.png').getpalette()

def convertion(masks, test_path,palette,info):
    _, C, T, H, W = masks.size()
    vid = np.zeros((1, 1, T, H, W))
    masks = torch.cat((1 - masks, masks), dim=1)
    for f in range(T):
        E = masks[0, :, f].cpu().data.numpy()
        # make hard label
        E = ToLabel(E)

        (lh, uh), (lw, uw) = info['pad']
        E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        img_E = Image.fromarray(E)
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

def Propagate_MS(ms, F2, P2):
    h, w = F2.size()[2], F2.size()[3]

    msv_F2, msv_P2 = ToCudaVariable([F2, P2])
    # r5, r4, r3, r2 = model.module.Encoder(msv_F2, msv_P2)
    # e2 = model.module.Decoder(r5, ms, r4, r3, r2)
    # r5, r4, r3, r2 = model.module.Encoder(msv_F2, msv_P2)
    # e2 = model.module.Decoder(r5, ms, r4, r3, r2)
    r5, r4, r3, r2 = model.Encoder(msv_F2, msv_P2)
    e2 = model.Decoder(r5, ms, r4, r3, r2)

    return F.softmax(e2[0], dim=1)[:, 1], r5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    use_crf = False

    testpath = '/mnt/gmy/pytorch/RGMP/Evaluation/save/'

    model = RGMP()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    Testset = DAVIS1(DAVIS_ROOT, imset='val.txt')
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=2)

    best_iou = 0
    writer = SummaryWriter()
    for epoch in np.sort([int(d.split('.')[0]) for d in os.listdir('saved_models')]):
        d = '{}.pth'.format(epoch)
        # load saved model if specified
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, d, font.END))
        load_name = os.path.join('saved_models',
                                 '{}.pth'.format(epoch))
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        model.load_state_dict(state)
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        del checkpoint
        torch.cuda.empty_cache()
        print('  - complete!')

        criterion = torch.nn.BCELoss()

        # testing
        with torch.no_grad():
            print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
            model.eval()
            loss = 0
            iOU = 0
            # F_a = 0
            pbar = tqdm.tqdm(total=len(Testloader))
            for i, (all_F, all_M, info) in enumerate(Testloader):
                pbar.update(1)
                all_F, all_M = all_F[0], all_M[0]
                seq_name = info['name'][0]
                num_frames = info['num_frames'][0]
                num_objects = info['num_objects'][0]

                test_path = os.path.join(testpath,seq_name)
                if not os.path.exists(test_path):
                    os.makedirs(test_path)

                B, C, T, H, W = all_M.shape
                all_E = torch.zeros(B, C, T, H, W)
                all_E[:, 0, 0] = all_M[:, :, 0]

                msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:, :, 0], all_E[:, 0, 0], all_M])
                ms = model.Encoder(msv_F1, msv_P1)[0]

                for f in range(0, all_M.shape[2] - 1):
                    output, ms = Propagate_MS(ms, all_F[:, :, f + 1], all_E[:, 0, f])
                    output1 = output
                    all_E[:, 0, f + 1] = output.detach()
                    loss = loss + criterion(output.permute(1, 2, 0), all_M[:, 0, f + 1].float()) / all_M.size(2)
                full_mask = torch.cat((1 - all_E, all_E), dim=1)
                if use_crf:
                    full_mask = dense_crf(np.array(all_F).astype(np.uint8), full_mask)
                iOU = iOU + iou(full_mask, all_M)
                # output1 = convertion(all_E,test_path,palette,info)


            pbar.close()

            loss = loss / len(Testloader)
            iOU = iOU / len(Testloader)
            # F_a = F_a/len(Testloader)
            writer.add_scalar('Val/BCE', loss, epoch)
            writer.add_scalar('Val/IOU', iOU, epoch)
            print('loss: {}'.format(loss))
            print('IoU: {}'.format(iOU))
            # print('F is:{}'.format(F))

            if best_iou < iOU:
                best_iou = iOU