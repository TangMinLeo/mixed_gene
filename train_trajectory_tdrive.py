from modules import *
import os
import argparse
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from decode_traj import CoordinateDecoder
from sklearn.preprocessing import QuantileTransformer
import pandas as pd



class ZipDataset(Dataset):
    def __init__(self, *datasets):
        """
        *datasets: 一个或多个 Dataset 实例
        """
        # 确保所有数据集的长度相同
        self.lengths = [len(d) for d in datasets]
        if len(set(self.lengths)) != 1:
            raise ValueError(f"所有数据集的长度必须相同，但得到了 {self.lengths}")
        
        self.datasets = datasets
        self.total_length = self.lengths[0]

    def __getitem__(self, index):
        # 从每个数据集中取出 index 对应的样本
        items = [ds[index] for ds in self.datasets]
        
        # 如果你想自动拼接 tensor 类型的特征：
        # 假设 items 中的元素都是 tensor，且你希望在第 0 维拼接
        # if all(isinstance(item, torch.Tensor) for item in items):
        #     return torch.cat(items, dim=0)
        
        # 更通用的做法：返回一个元组或字典
        # 如果是不同类型的样本（如图片和标签），返回元组更合适
        return tuple(items)

    def __len__(self):
        return self.total_length


class TrajectoryProcessor:
    def __init__(self, eps=0.5):
        self.eps = eps
        self.qt_d = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0)
        self.qt_dt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0)
    
    def fit(self, dx, dy, dt):
        d =np.sqrt(dx**2 + dy**2)
        print('d max: ', np.max(d), 'd min: ',  np.min(d))
        print('dt max: ', np.max(dt), 'dt min: ',  np.min(dt))
        
        d_thr = d[d > self.eps]
        print('d_thr max: ', np.max(d_thr), 'd_thr min: ',  np.min(d_thr))
        print('d_thr len: ', len(d_thr))
        self.qt_d.fit(d_thr.reshape(-1, 1))
        self.qt_dt.fit(dt.reshape(-1, 1))
        
        # custom_bins = list(range(0, 601))  # 从0到1000，每10一个区间
        # hist, bin_edges = np.histogram(dt, bins=custom_bins)
        # print(f"频数: {hist}")
        # print(f"边界: {bin_edges}")
        
    
    def transformer(self, dx, dy, dt):
        d = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        is_stationary = (d < self.eps)
        # zeros_ratio = np.mean(d < 1e-5)
        # print(f'比例：{zeros_ratio:.4f}，数量：{np.sum(d < 1e-5)}')
    
        d_transformed = self.qt_d.transform(d.reshape(-1, 1)).reshape(dx.shape)
        dt_transformed = self.qt_dt.transform(dt.reshape(-1, 1)).reshape(dt.shape)
        
        for i in range(d.shape[0]):
            for j in range(1, d.shape[1]):
                if is_stationary[i, j]:
                    theta[i, j] = theta[i, j-1]
                    # 真实的静止点就使用0分位对应的正态z值，而不是0
                    # d_transformed[i, j] = 0.0
        
        is_stationary = is_stationary.astype(np.float32)
        
        feature = np.stack([d_transformed, np.sin(theta)*(1-is_stationary), np.cos(theta)*(1-is_stationary), dt_transformed, is_stationary], axis=-1)
        feature[:, 0, :] = 0.0  # 将第0个点的特征设置为0，确保它们在训练中不产生影响，保证轨迹起点和填充点的都为0，真正的静止点使用0分位对应的正态z值
        
        return feature
    
    def inverse_transformer(self, feature):
        is_stationary = feature[..., -1] > 0.5
        
        d = self.qt_d.inverse_transform(feature[..., 0].reshape(-1, 1)).reshape(feature.shape[:2])
        dt = self.qt_dt.inverse_transform(feature[..., 3].reshape(-1, 1)).reshape(feature.shape[:2])
        
        theta = np.arctan2(feature[..., 1], feature[..., 2])
        
        dx = d * np.cos(theta)*(1-is_stationary)
        dy = d * np.sin(theta)*(1-is_stationary)
        
        return dx, dy, dt, is_stationary
        

def launch(data_loader):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_trajectory_v_pre_10"

    if os.path.exists(f'./training/{args.run_name}'):
        pass
    else:
        os.makedirs(f'./training/{args.run_name}')

    args.in_c = 128
    # args.out_c = 5200

    args.emb_dim = 5
    # args.emb_dim = 164

    args.epochs = 1500
    # args.epochs = 455
    
    args.batch_size = 48
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args.lr = 3e-4
    # args.lr = 0.00006293
    
    args.save_file = f'./training/{args.run_name}/training_process_v_pre_1500_epochs.json'
    # args.new_values = new_values
    # args.probabilities = probabilities

    # train(args, data_loader, actual_len, decoder)
    train(args, data_loader)


def train(args, dataloader):
    # for param in decoder.parameters():
    #     param.requires_grad = False
    
    device = args.device
    print(device)
    
    model = Net_model(c_in=args.in_c, c_out=args.in_c, emb_dim=512, first_tim_dim=8, first_pos_dim=128, device="cuda").to(device)
    # checkpoint = torch.load('./training/DDPM_trajectory_v_pre_8/ckpt_1045.pt')
    # model.load_state_dict(checkpoint)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    # mse = torch.nn.MSELoss()

    diffusion = Diffusion(noise_steps=2000, beta_start=1e-4, beta_end=0.02, channels=args.in_c, emb_dim=args.emb_dim, device=device)

    # l = len(dataloader[0])
    # print('dataloader len: ', l)

    # ff = open(f'./voca_map.json', 'r', encoding='utf-8')
    # voc_map = json.load(ff)
    # ff.close()

    # data_map = {}
    # for key, value in voc_map.items():
    #     data_map[str(int(value))] = key

    gene_dict = {}
    ff = open(args.save_file, "w")
    
    logging.basicConfig(
    filename=f'./training/{args.run_name}/training_process_v_pre_1500_epochs.log',  # 每次写入都会追加到文件末尾
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
    

    # start_epoch = 1046
    start_epoch = 0
    
    for epoch in range(start_epoch, args.epochs + start_epoch):
        print(f'starting epoch {epoch}:')
        # pbar = tqdm(dataloader)
        loss_list = []
        # for i, samples in enumerate(pbar):
        
        # for samples, batch_mask in dataloader:
        #     samples = samples.to(device)
        #     batch_mask = batch_mask.to(device)
        #     # 强制让 samples 的填充位为 0，防止脏数据干扰加噪
        #     samples = samples * (~batch_mask).unsqueeze(-1)
        cou = 0
        for batch in dataloader:
            # combined = np.concatenate([emb_posi_seq, norm_emb_posi_seq_d, norm_emb_time_seq, norm_abs_emb_time, norm_abs_emb_time_seq], axis=-1)
            cou += 1
            # combined, start_time_emb, emb_posi_seq, mask_= batch
            combined, start_time_emb, emb_start_point_seq, progress_seq = batch
            
            combined = combined.to(device).float()
            start_time_emb = start_time_emb.to(device).float()
            # emb_start_point_seq = emb_start_point_seq.to(device)
            progress_seq = progress_seq.to(device).float()
            
            # mask_ = mask_.to(device)
            # combined = combined * (~mask_).unsqueeze(-1)
            
            start_pos_emb = emb_start_point_seq.to(device).float()  # 将 emb_posi_seq 转换为张量并移动到设备上
            
            # start_pos_emb = combined[:, :1, :128]  # 提取起始位置的编码
            # combined = combined[:, :, 128:]  # 去掉起始位置的编码，剩下的作为输入特征
            
            # print('samples.shape: ', samples.shape)
            # samples = samples + 0.002*torch.randn(samples.shape, device=device)
            # samples = torch.clamp(samples, 0.0, 1.0)
            t = diffusion.sample_timesteps(combined.shape[0]).to(device)
            # print('combined example: ', combined[:3, :5, :6])
            x_t, noise = diffusion.noise_images(combined, t)
            # print(mask_[0, :20])
            
            # 同样确保 x_t 的填充位在进入模型前也是干净的
            # x_t = x_t * (~mask_).unsqueeze(-1)
            # print('x_t dtype: ', x_t.dtype)
            # print('mask_ dtype: ', mask_.dtype)
            # print('t dtype: ', t.dtype)
            # print('start_time_emb dtype: ', start_time_emb.dtype)
            # print('start_pos_emb dtype: ', start_pos_emb.dtype)
            # print(f't after use in adding noise is on cuda: {t.is_cuda}')
            # predicted_noise = model(x_t, t)
            
            x_t = x_t.float()
            start_flag_idx = torch.zeros([combined.shape[0], combined.shape[1]], dtype=torch.int32, device=device)

            # start_flag_idx = torch.zeros_like(mask_, dtype=torch.int32).to(device)  # 创建与 mask_ 第一列相同形状的张量
            start_flag_idx[:, 0] = 1
            
            # predicted_v = model(x_t, t, mask_, start_time_emb, start_pos_emb, start_flag_idx)
            predicted_v = model(x_t, t, start_time_emb, start_pos_emb, start_flag_idx, progress_seq)
            
            # if cou >300:
            #     print('predicted_v sample: ', predicted_v[:3, :5, :7])
            
            v_t = torch.sqrt(diffusion.alpha_hat[t][:, None, None])*noise - torch.sqrt(1-diffusion.alpha_hat[t][:, None, None])*combined
            # if cou >300:
            #     print('v_t sample: ', v_t[:3, :5, :7])
                
            # print('v_t dtype: ', v_t.dtype)
            # print(predicted_noise.dtype)
            # 模型用来拟合v_t
            # loss = mse(v_t, predicted_v)
            
            loss_ele_wise = F.mse_loss(v_t, predicted_v, reduction='none')
            # print('loss_ele_wise dtype: ', loss_ele_wise.dtype)
            # print('loss_ele_wise.shape: ', loss_ele_wise.shape)
            
            # weights = torch.ones_like(loss_ele_wise, requires_grad=False)
            # weights[:, 0, :] *= 5.0  # 给第0个点（起始时间+位置）5倍权重
            # weights = weights.to(device)
            
            weights = torch.ones_like(loss_ele_wise, requires_grad=False)
            
            weights[:, 0, :] = 0.0  # 将第0个点不参与损失计算
            # weights[:, :, :2] *= 10.0  # 给相对位置误差20倍权重
            weights[:, :, 3] *= 5.0      # 给相对时间误差10倍权重
            weights = weights.to(device).float()
            # print('weights dtype: ', weights.dtype)
            
            # 只计算非 Padding 部分的 Loss 总和，并且乘以权重
            # active_loss = (loss_ele_wise * weights * (~mask_).unsqueeze(-1)).sum()
            active_loss = (loss_ele_wise * weights).sum()
            
            # print('active_loss type: ', active_loss.dtype)
            
            # 只除以有效元素的个数，而不是 batch_size * seq_len * feature_dim
            # actual_num_elements = ((~mask_).sum() - mask_.size(0)) * loss_ele_wise.shape[-1]
            actual_num_elements = weights.shape[0] * (weights.shape[1] - 1) * loss_ele_wise.shape[-1]
            # print('actual_num_elements type: ', actual_num_elements.dtype)
            loss = active_loss / (actual_num_elements + 1e-8) # 防止除以0
        
            # print(loss.dtype)
            
            # 忽略 Padding 部分
            # loss = (loss_ele_wise * weights * (~batch_mask).unsqueeze(-1)).mean()
            
            # loss = mse(noise, predicted_noise)

            # print(loss.dtype)
            optimizer.zero_grad()
            loss.backward()
            
            # x0 = torch.cat([m_feat 128, t_feat 128, f_feat 256], dim=-1)
            if cou % 600 == 0 and epoch % 50 == 0:  # 每 100 个 batch 检查一次
                check_input_layer_gradients(model, flag_dim=384, physics_dim=128)
            
            optimizer.step()

            copy_loss = loss.clone()
            # copy_loss = copy_loss.detach().item()
            # pbar.set_postfix(MSE=copy_loss.item())
            # pbar.set_postfix(MSE=loss.item())

            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            # print("MSE", loss.item())

            loss_list.append(copy_loss.item())

        # sampled_items, sampled_mask = diffusion.sample(model, 1, args.new_values, args.probabilities)
        # print(sampled_items)
        # print(sampled_mask)
        
        # decoded_seqs, decoded_tims = decode(sampled_items.tolist(), sampled_mask.tolist())
        # print('decoded_seqs: ', decoded_seqs[:5])
        # print('decoded_tims: ', decoded_tims[:5])
        
        
        
        
        # data = pd.DataFrame(sampled_items, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

        # u_dist = copula.u_dist
        # # mapping1 = copula.reverse_mapping
        # all_row = []
        # for i in range(len(data)):
        #     one_row = []
        #     for col in data.columns:
        #         key = col
        #         series = pd.Series(u_dist[key])
        #         one_value = data.iloc[i][col]
        #         idx = find_nearest(series, one_value)
        #         one_row.append(series[idx])
        #         # one_row.append(data_map[str(int(mapping1[key][int(idx)]))])
        #     # print(one_row)
        #     all_row.append(one_row)

        # total_row = pd.DataFrame(all_row, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

        # inversed_cdf = copula.inverse_transform(total_row)

        # sample_out = inversed_cdf.values.tolist()
        # # gene_rep = []
        # for j in range(len(sample_out)):
        #     tem = []
        #     for item in sample_out[j]:
        #         tem.append(data_map[str(int(item))])
        #     # gene_rep.append(tem)
        #     print(tem)
        
        
        # gene_dict[epoch] = [gene_rep, np.mean(np.array(loss_list)), loss_list]
        # gene_dict[epoch] = [sample_out, np.mean(np.array(loss_list)), loss_list]
        
        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("training", args.run_name, f"ckpt_{epoch}.pt"))

        scheduler.step()
        # print(f'learning rate: {scheduler.get_last_lr()}')
        gene_dict[epoch] = [np.mean(np.array(loss_list)), loss_list, scheduler.get_last_lr()]
        # gene_dict[epoch] = [sample_out, np.mean(np.array(loss_list)), loss_list, scheduler.get_last_lr()]
        ave_loss = float(np.mean(np.array(loss_list)))
        
        print('epoch: ', epoch, 'average loss: ', ave_loss)
        
        lr = float(scheduler.get_last_lr()[-1])
        logging.info(f"Epoch {epoch}: loss={ave_loss:.8f}, learning_rate={lr:.8f}")

    json.dump(gene_dict, ff, ensure_ascii=False)
    ff.close()


if __name__ == "__main__":
    # day_traces.json 包含154113条数据
    # day_traces_1.json 包含2954290条数据
    # josn_file = './day_traces_1.json'
    # time_seq, posi_seq = get_original_data(josn_file)
    
    f = open('./trips_posi_seq.json', 'r')
    trips_posi_seq = json.load(f)
    f.close()

    f = open('./trips_time_seq.json', 'r')
    trips_time_seq = json.load(f)
    f.close()
    
    resampled_trajectory = []
    start_time = []
    
    for i in range(len(trips_time_seq)):
        one_time_seq = trips_time_seq[i]
        one_posi_seq = trips_posi_seq[i]
        
        one_data = pd.DataFrame({
            'time': pd.to_datetime(one_time_seq, unit='s'),
            'lng': [p[0] for p in one_posi_seq],
            'lat': [p[1] for p in one_posi_seq]
        })
        
        start_time.append(one_time_seq[0])
        
        resampled_tensor = resample_trip_to_fixed_N(one_data, target_N=128)
        
        # print(resampled_tensor[:10,:])
        
        resampled_trajectory.append(resampled_tensor)
    
    emb_start_point_seq, emb_posi_seq_d, start_time_emb, emb_time_seq, progress_seq = make_features(resampled_trajectory, start_time)
        
    print(emb_start_point_seq.shape)   
    print(emb_posi_seq_d.shape)
    print(start_time_emb.shape)
    print(emb_time_seq.shape)
    print(progress_seq.shape)
    
    # seq_l = 422
    # max_T = 20000
    # max_T = 30587
    # emb_posi_seq, emb_posi_seq_d, emb_time_seq, abs_emb_time_seq, mask_ = construct_features(time_seq[:10], posi_seq[:10], 520)
    # emb_posi_seq, emb_posi_seq_d, emb_time_seq, abs_emb_time, abs_emb_time_seq, start_time_emb, mask_ = construct_features(v_thre_time_seq, v_thre_posi_seq, 216)
    # emb_posi_seq, emb_posi_seq_d, emb_time_seq, start_time_emb, mask_ = construct_features(v_thre_time_seq, v_thre_posi_seq, seq_l)
    
    all_dx = []
    all_dy = []
    all_dt = []

    trace_count = 0
    for i in range(emb_posi_seq_d.shape[0]):
        ds_dt_count = 0
        # 不考虑第一个点，因为第一个点是轨迹的起点被设置为全0，没有意义
        for j in range(1, emb_posi_seq_d.shape[1]):
            x = emb_posi_seq_d[i, j, 0]
            y = emb_posi_seq_d[i, j, 1]
            all_dx.append(x)
            all_dy.append(y)
            all_dt.append(emb_time_seq[i, j, 0])
    
    
    traj_proc = TrajectoryProcessor(eps=0.1)
    # fit的时候要拉成一个一维长向量
    # transforme的时候要保持原来的形状，最后输出的feature也是和输入形状一样的
    traj_proc.fit(np.array(all_dx), np.array(all_dy), np.array(all_dt))
    
    feat = traj_proc.transformer(emb_posi_seq_d[:, :, 0], emb_posi_seq_d[:, :, 1], np.array(emb_time_seq)[:, :, 0])
    print('feat shape: ', feat.shape)
    
    # 已经经过重采样使得轨迹具有相同的长度，所以不需要再对mask_进行处理了
    # feat[mask_] = 0.0  # 将填充部分的特征设置为0，确保它们在训练中不产生影响
    
    print(feat[23, :32, :])
    
    
    # lengths = np.array([len(seq)-sum(seq) for seq in mask_])
    # print(lengths[:100])
    
    # sorted_indices = np.argsort(lengths)
    # # np.argsort返回的是下标，即把lengths数组从小到大排序后对应的原始下标。sorted_indices[0]是最短轨迹的下标，sorted_indices[-1]是最长轨迹的下标。
    # lengths_sorted = lengths[sorted_indices]
    # print('lengths_sorted: ', lengths_sorted[:10])
    
    # print(sorted_indices[0])
    # # print(norm_emb_time_seq[sorted_indices[0]])
    
    # # 去重后95%分位数对应的轨迹长度比不去重的更长，实际上重复长度的轨迹应该计算数量。去重后0.95分位数对应的轨迹长度是10349，而不去重的0.95分位数对应的轨迹长度是5204。
    # i = int(len(lengths)*0.95)+1
    # print(lengths_sorted[i])
    
    # values, counts = np.unique(lengths_sorted, return_counts=True)
    # # plt.bar(values, counts)
    
    # new_values = []
    # new_counts = []
    
    # for value, count in zip(values, counts):
    #     if value <= seq_l:
    #         new_values.append(value)
    #         new_counts.append(count)
    # new_values = np.array(new_values)
    # new_counts = np.array(new_counts)
    # print('len(new_values) ', len(new_values))
    # # plt.plot(new_values, new_counts)
    # # plt.show()
    
    # probabilities = new_counts / new_counts.sum()
    # print('sum_probability ', probabilities.sum())
    
    # print('new_values: ', new_values)
    # print('probabilities: ', probabilities)
    # sample = np.random.choice(new_values, size=100, p=probabilities)
    # print(sample)
    
    
    # combined = np.concatenate([norm_emb_posi_seq_d, norm_emb_time_seq, norm_abs_emb_time, norm_abs_emb_time_seq], axis=-1)
    
    # dataset = CustomDataset(combined, mask_)
    # dataloader = DataLoader(dataset, batch_size=48, shuffle=True, drop_last=False)
    
    # combined shape b*216*164
    # combined = np.concatenate([norm_emb_posi_seq_d, norm_emb_time_seq], axis=-1)
    combined = feat
    
    # combined = np.concatenate([emb_posi_seq, norm_emb_posi_seq_d, norm_emb_time_seq, norm_abs_emb_time, norm_abs_emb_time_seq], axis=-1)
    # combined = np.concatenate([emb_posi_seq, norm_emb_posi_seq_d, norm_emb_time_seq, norm_abs_emb_time_seq], axis=-1)
    print('combined.shape: ', combined.shape)
    
    # b = combined.shape[0]
    # seq_l = combined.shape[1]
    
    # # 创建 [L, 1] 的 flag，第一点为 1，其他点为 0
    # start_flag = np.zeros((b, seq_l, 1), dtype=np.float32)
    # start_flag[:, 0, 0] = 1.0  # 仅第一点为 1
    
    # # 拼接后特征维度变为 D + 1
    # combined = np.concatenate([start_flag, combined], axis=-1)
    # print('combined.shape: ', combined.shape)
    

    zipped_dataset = ZipDataset(combined, start_time_emb, emb_start_point_seq, progress_seq)
    dataloader = DataLoader(zipped_dataset, batch_size=128, shuffle=True, drop_last=False)
    
    # sta_dict = torch.load('./coord_decoder.pt')
    # decoder = CoordinateDecoder(input_dim=128, hidden_dim=512, num_blocks=2)
    # decoder.load_state_dict(state_dict=sta_dict)
    
    launch(dataloader)
    