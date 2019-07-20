import torch
import argparse, threading, time
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
from tensorboardX import SummaryWriter
import datetime
from IPython import embed

# Constants
MODEL_DIR = 'saved_models_test'
NUM_EPOCHS = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

palette = Image.open('/mnt/gmy/Tensorflow/video_seg/DAVIS2017/DAVIS-2017-trainval-480p/Annotations/480p/bear/00000.png').getpalette()

def convertion(masks, test_path,palette):
    _, C, T, H, W = masks.size()
    vid = np.zeros((1, 1, T, H, W))
    masks = torch.cat((1 - masks, masks), dim=1)
    for f in range(T):
        E = masks[0, :, f].cpu().data.numpy()
        # make hard label
        E = ToLabel(E)
        img_E = Image.fromarray(E)
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

def Propagate_MS(ms, F2, P2):
    h, w = F2.size()[2], F2.size()[3]

    msv_F2, msv_P2 = ToCudaVariable([F2, P2])
    r5, r4, r3, r2 = model.module.Encoder(msv_F2, msv_P2)
    e2 = model.module.Decoder(r5, ms, r4, r3, r2)
    # r5, r4, r3, r2 = model.Encoder(msv_F2, msv_P2)
    # e2 = model.Decoder(r5, ms, r4, r3, r2)

    return F.softmax(e2[0], dim=1)[:, 1], r5


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='RGMP')
    parser.add_argument('--epochs', dest='num_epochs',
                        help='number of epochs to train',
                        default=NUM_EPOCHS, type=int)
    parser.add_argument('--bs', dest='bs',
                        help='batch_size',
                        default=6, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='display interval',
                        default=10, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                        help='interval of epochs to perform validation',
                        default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory',
                        default=MODEL_DIR, type=str)

    # BPTT
    parser.add_argument('--bptt', dest='bptt_len',
                        help='length of BPTT',
                        default=12, type=int)
    parser.add_argument('--bptt_step', dest='bptt_step',
                        help='step of truncated BPTT',
                        default=4, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--alpha', dest='alpha',
                        help='alpha in the RMSprop', default=0.9,
                        type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum in RMSprop',
                        default=0.5, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay in RMSprop',
                        default=1.0, type=float)
    parser.add_argument('--eps', dest='eps',
                        help='eps in RMSprop',
                        default=1.0, type=float)
    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch',
                        help='epoch to load model',
                        default=945, type=int)

    args = parser.parse_args()
    return args


def log_mask(frames, masks, info, writer, F_name='Train/frames', M_name='Train/masks'):
    print('[tensorboard] Updating mask..')
    _, C, T, H, W = masks.size()
    #     (lh,uh), (lw,uw) = info['pad']
    vid = np.zeros((1, 3, T, H, W))
    print_dbg('[mask] mean: {}, max: {}'.format(torch.mean(masks[:, :, 1]), torch.max(masks[:, :, 1])))
    masks = torch.cat((1 - masks, masks), dim=1)
    for f in range(T):
        E = masks[0, :, f].cpu().data.numpy()
        # make hard label
        E = ToLabel(E)
        #         E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        # need to implement mask overlay

        img_E = Image.fromarray(E)
        img_E.putpalette(PALETTE)
        arr_E = np.array(E)
        vid[0, :, f, :, :] = np.array(img_E)

    vid_tensor = torch.FloatTensor(vid)
    # embed()
    writer.add_video(F_name, vid_tensor=frames)
    writer.add_video(M_name, vid_tensor=vid_tensor)
    print('[tensorboard] Mask updated')



if __name__ == '__main__':

    test_path = '/mnt/gmy/pytorch/RGMP/save1/'

    args = parse_args()
    Trainset = DAVIS(DAVIS_ROOT, imset='2017/train.txt')
    Trainloader = data.DataLoader(Trainset, batch_size=1, shuffle=True, num_workers=2)

    Testset = DAVIS(DAVIS_ROOT, imset='2017/val.txt')
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=2)

    model = RGMP()
    # if torch.cuda.is_available():
    #     model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    writer = SummaryWriter()
    start_epoch = 0

    # load saved model if specified
    if args.loadepoch >= 0:
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, args.loadepoch, font.END))
        load_name = os.path.join(args.output_dir,
                                 '{}.pth'.format(args.loadepoch))
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

    # params
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value], 'lr': args.lr, 'weight_decay': 4e-5}]


    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.5, weight_decay=0.5)


    iters_per_epoch = len(Trainloader)
    for epoch in range(start_epoch, args.num_epochs):
        if epoch % args.eval_epoch == 1:
            # testing
            with torch.no_grad():
                print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
                model.eval()
                loss = 0
                iOU = 0
                pbar = tqdm.tqdm(total=len(Testloader))
                for i, (all_F, all_M, info) in enumerate(Testloader):
                    pbar.update(1)
                    all_F, all_M = all_F[0], all_M[0]
                    seq_name = info['name'][0]
                    num_frames = info['num_frames'][0]
                    num_objects = info['num_objects'][0]

                    B, C, T, H, W = all_M.shape
                    all_E = torch.zeros(B, C, T, H, W)
                    all_E[:, 0, 0] = all_M[:, :, 0]

                    msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:, :, 0], all_E[:, 0, 0], all_M])
                    ms = model.module.Encoder(msv_F1, msv_P1)[0]

                    for f in range(0, all_M.shape[2] - 1):
                        output, ms = Propagate_MS(ms, all_F[:, :, f + 1], all_E[:, 0, f])
                        all_E[:, 0, f + 1] = output.detach()
                        loss = loss + criterion(output.permute(1, 2, 0), all_M[:, 0, f + 1].float()) / all_M.size(2)
                    iOU = iOU + iou(torch.cat((1 - all_E, all_E), dim=1), all_M)


                pbar.close()

                loss = loss / len(Testloader)
                iOU = iOU / len(Testloader)
                writer.add_scalar('Val/BCE', loss, epoch)
                writer.add_scalar('Val/IOU', iOU, epoch)
                print('val loss: {},time:{}'.format(loss, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                print('val IoU: {}'.format(iOU))

                video_thread = threading.Thread(target=log_mask,
                                                args=(all_F, all_E, info, writer, 'Val/frames', 'Val/masks'))
                video_thread.start()

        # training
        model.train()
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
        for i, (all_F, all_M, info) in enumerate(Trainloader):
            optimizer.zero_grad()
            all_F, all_M = all_F[0], all_M[0]
            seq_name = info['name'][0]
            num_frames = info['num_frames'][0]
            num_objects = info['num_objects'][0]

            if (args.bptt_len < num_frames):
                start_frame = random.randint(0, num_frames - args.bptt_len)
                all_F = all_F[:, :, start_frame: start_frame + args.bptt_len]
                all_M = all_M[:, :, start_frame: start_frame + args.bptt_len]

            tt = time.time()

            B, C, T, H, W = all_M.shape
            all_E = torch.zeros(B, C, T, H, W)
            all_E[:, 0, 0] = all_M[:, :, 0]

            msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:, :, 0], all_E[:, 0, 0], all_M])
            ms = model.module.Encoder(msv_F1, msv_P1)[0]

            num_bptt = all_M.shape[2]
            loss = 0
            counter = 0
            for f in range(0, num_bptt - 1):
                output, ms = Propagate_MS(ms, all_F[:, :, f + 1], all_E[:, 0, f])
                all_E[:, 0, f + 1] = output.detach()
                loss = loss + criterion(output.permute(1, 2, 0), all_M[:, 0, f + 1].float())
                counter += 1
                if (f + 1) % args.bptt_step == 0:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    output.detach()
                    if f < num_bptt - 2:
                        loss = 0
                        counter = 0
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # logging and display
            if (i + 1) % args.disp_interval == 0:
                writer.add_scalar('Train/BCE', loss / counter, i + epoch * iters_per_epoch)
                writer.add_scalar('Train/IOU', iou(torch.cat((1 - all_E, all_E), dim=1), all_M),
                                  i + epoch * iters_per_epoch)
                print('{} train loss: {},time is:{}'.format(i , loss / counter,
                                                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            if epoch % 10 == 1 and i == 0:
                video_thread = threading.Thread(target=log_mask, args=(all_F, all_E, info, writer))
                video_thread.start()

        if epoch % 5 == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                       save_name)


