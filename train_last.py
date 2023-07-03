import os

import torch
from tqdm.auto import tqdm
from opt import config_parser
import matplotlib.pyplot as plt
import math
import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys
from models.discriminator import ResNetDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


# class SimpleSampler:
#     def __init__(self, total, batch):
#         self.total = total
#         self.half = math.sqrt(total)
#         self.batch = batch
#         self.curr = total
#         self.ids = None
#
#     def nextids(self):
#         self.curr+=self.batch
#         if self.curr + self.batch > self.total:
#             self.ids = torch.LongTensor(np.random.permutation(self.total))
#             self.curr = 0
#         return self.ids[self.curr:self.curr+self.batch]
class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]

class GridSampler:
    def __init__(self, total, batch, dilated=4):
        self.total = total
        self.batch = batch
        self.d = dilated
        self.H = int(math.sqrt(total / 100))
        self.h = int(math.sqrt(batch))
        self.ids = torch.LongTensor(np.arange(self.total))
        self.horizontal_boundary = self.H - (self.h + self.d * (self.h - 1))
        self.vertical_boundary = self.horizontal_boundary * (self.H + 1)
        if self.horizontal_boundary <= 0:
            raise Exception(self.horizontal_boundary)

    def nextids(self):
        img_num = np.random.randint(1, 100)
        topleft = np.random.randint(1, self.total)
        while (topleft % self.H > self.horizontal_boundary or topleft > self.vertical_boundary):
            topleft = np.random.randint(1, self.total)
        sampled_start_rays = [i * (self.d + 1) * self.H + topleft for i in range(self.h)]
        sampled_rays = np.array([i * (self.d + 1) + a for a in sampled_start_rays for i in range(self.h)])
        # print(len(sampled_rays))
        sampled_rays += img_num * self.H ** 2
        return self.ids[sampled_rays]


# a=GridSampler(49,9,1)
# a.nextids()

@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                   N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                                        density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                                        app_dim=args.data_dim_color, near_far=near_far,
                                        shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                        density_shift=args.density_shift, distance_scale=args.distance_scale,
                                        pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                        featureC=args.featureC, step_ratio=args.step_ratio,
                                        fea2denseAct=args.fea2denseAct)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    #  discriminator
    discriminator = ResNetDiscriminator().to(device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.99), lr=0.0001)

    # linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(
        torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[
                   1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    dilated = [20, 20, 20, 20, 20]
    dilated_index = 0
    allrays_simple, allrgbs_simple = train_dataset.all_rays, train_dataset.all_rgbs
    allrays_grid, allrgbs_grid = train_dataset.all_rays, train_dataset.all_rgbs

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays_simple, allrgbs_simple, bbox_only=True)
    simpleSampler = SimpleSampler(allrays_simple.shape[0], args.batch_size)
    gridSampler = GridSampler(allrays_simple.shape[0], args.batch_size, dilated[dilated_index])  # GridSampler로 교체

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx_simple = simpleSampler.nextids()
        ray_idx_grid = gridSampler.nextids()
        rays_train_simple, rgb_train_simple = allrays_simple[ray_idx_simple], allrgbs_simple[ray_idx_simple].to(device)
        rays_train_grid, rgb_train_grid = allrays_grid[ray_idx_grid], allrgbs_grid[ray_idx_grid].to(device)

        # for discriminator loss
        rgb_map_fake, alphas_map_fake, depth_map_fake, weights_fake, uncertainty_fake \
            = renderer(rays_train_grid, tensorf, chunk=args.batch_size,
                       N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        # for L2 loss
        # rgb_map, alphas_map, depth_map, weights, uncertainty \
        #     = renderer(rays_train_grid, tensorf, chunk=args.batch_size,
        #                N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        # discriminator loss
        h = int(math.sqrt(rgb_train_grid.shape[0]))
        real_predictions = discriminator(rgb_train_grid.reshape(1, 3, h, h))
        fake_predictions = discriminator(rgb_map_fake.reshape(1, 3, h, h).detach())

        real_labels = torch.ones(real_predictions.size()).to(device)
        fake_labels = torch.zeros(fake_predictions.size()).to(device)
        discriminator_loss = adversarial_loss(real_predictions, real_labels) \
                             + adversarial_loss(fake_predictions, fake_labels)

        discriminator_loss.backward()
        discriminator_optimizer.step()

        if not (iteration % 10):
            # generator loss
            rgb_map_fake_grid, alphas_map_fake_grid, depth_map_fake, weights_fake, uncertainty_fake \
                = renderer(rays_train_grid, tensorf, chunk=args.batch_size,
                           N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

            rgb_map_fake_simple, alphas_map_fake_simple, depth_map_fake, weights_fake, uncertainty_fake \
                = renderer(rays_train_simple, tensorf, chunk=args.batch_size,
                           N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

            fake_predictions_generator = discriminator(rgb_map_fake_grid.reshape(1, 3, h, h))
            generator_loss = adversarial_loss(fake_predictions_generator, real_labels)

            # if iteration and not (iteration % 1000):
            #     fig = plt.figure()
            #     rows, cols = 1, 2
            #
            #     ax1 = fig.add_subplot(rows, cols, 1)
            #     ax1.imshow(rgb_map.reshape(32, 32, 3).cpu().detach())
            #     ax1.set_title('Rendered patch')
            #     ax1.axis("off")
            #
            #     ax2 = fig.add_subplot(rows, cols, 2)
            #     ax2.imshow(rgb_train.reshape(32, 32, 3).cpu().detach())
            #     ax2.set_title('GT')
            #     ax2.axis("off")
            #
            #     plt.show()

            aux_loss, img_loss = 0, 0

            if Ortho_reg_weight > 0:
                loss_reg = tensorf.vector_comp_diffs()
                aux_loss += Ortho_reg_weight * loss_reg
                summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
            if L1_reg_weight > 0:
                loss_reg_L1 = tensorf.density_L1()
                aux_loss += L1_reg_weight * loss_reg_L1
                summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

            if TV_weight_density > 0:
                TV_weight_density *= lr_factor
                loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                aux_loss += loss_tv
                summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)

            if TV_weight_app > 0:
                TV_weight_app *= lr_factor
                loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
                aux_loss += loss_tv
                summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

            img_loss = torch.mean((rgb_map_fake_simple - rgb_train_simple) ** 2)
            total_loss = img_loss + generator_loss + aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = total_loss.detach().item()

            PSNRs.append(-10.0 * np.log(img_loss.cpu().detach()) / np.log(10.0))
            # PSNRs.append(-10.0 * np.log(img_loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse', loss, global_step=iteration)
            summary_writer.add_scalar('train/generator_loss', generator_loss, global_step=iteration)
            summary_writer.add_scalar('train/discriminator_loss', discriminator_loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
                + f' dis_loss = {discriminator_loss:.3f}'
                + f' gen_loss= {generator_loss:.3f}'
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256 ** 3:  # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays_simple, allrgbs_simple = tensorf.filtering_rays(allrays_simple, allrgbs_simple)
                simpleSampler = SimpleSampler(allrgbs_simple.shape[0], args.batch_size)
        # if iteration and iteration % 10000 == 0:
        #     dilated_index += 1
        #     trainingSampler = GridSampler(allrays.shape[0], args.batch_size, dilated[dilated_index])

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>', c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

