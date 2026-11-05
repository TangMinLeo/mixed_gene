from modules import *
import os
import argparse
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from decode_traj import CoordinateDecoder
from sklearn.preprocessing import QuantileTransformer
import joblib


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
        
        feature = np.stack([d_transformed, np.sin(theta)*(1-is_stationary), np.cos(theta)*(1-is_stationary), dt_transformed], axis=-1)
        
        # feature = np.stack([d_transformed, np.sin(theta)*(1-is_stationary), np.cos(theta)*(1-is_stationary), dt_transformed, is_stationary], axis=-1)
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


class DifferentiableInverseQuantile(nn.Module):
    def __init__(self, sklearn_scaler, output_distribution='normal'):
        """
        修正版：将 sklearn 的 QuantileTransformer 逆变换完全平移至 PyTorch
        """
        super().__init__()
        self.output_distribution = output_distribution
        
        # 1. 提取物理距离网格 (Y轴)
        # sklearn_scaler.quantiles_ 存储的是原始数据(物理距离)按从小到大排序的锚点
        physical_distances = sklearn_scaler.quantiles_[:, 0] 
        
        # 2. 构建概率网格 (X轴)
        # 无论目标是 normal 还是 uniform，逆变换时我们都需要在 [0, 1] 的均匀概率空间中进行插值寻址
        n_quantiles = sklearn_scaler.n_quantiles_
        prob_grid = np.linspace(0, 1, n_quantiles)
        
        # 注册为 Buffer，全程参与 GPU 运算且不更新梯度
        self.register_buffer('prob_grid', torch.tensor(prob_grid, dtype=torch.float32))
        self.register_buffer('physical_grid', torch.tensor(physical_distances, dtype=torch.float32))

    def forward(self, q_pred):
        """
        q_pred: 模型预测出的隐空间变量 (如正态分布下的 QT(r))，形状为 [B, L]
        """
        # 1. 空间转换：将模型预测的高斯变量映射回 [0, 1] 概率空间
        if self.output_distribution == 'normal':
            # 高斯 CDF 变换 (可微)
            p_pred = 0.5 * (1.0 + torch.erf(q_pred / math.sqrt(2.0)))
        else:
            p_pred = q_pred
            
        # 裁剪防止越界
        p_pred = torch.clamp(p_pred, 1e-7, 1.0 - 1e-7)

        # 2. 可微的分段线性插值 (在 [0, 1] 网格上找对应的物理距离)
        shape = p_pred.shape
        p_flat = p_pred.contiguous().view(-1)
        
        # 使用 torch.bucketize 在 X轴 (prob_grid) 寻找左右桶索引
        indices = torch.bucketize(p_flat, self.prob_grid)
        indices = torch.clamp(indices, 1, len(self.prob_grid) - 1)
        
        # 获取插值区间的左右概率端点 (X坐标)
        p0 = self.prob_grid[indices - 1]
        p1 = self.prob_grid[indices]
        
        # 获取插值区间的左右物理距离端点 (Y坐标)
        r0 = self.physical_grid[indices - 1]
        r1 = self.physical_grid[indices]
        
        # 线性插值公式 (完美可微！)
        weights = (p_flat - p0) / (p1 - p0 + 1e-8)
        r_flat = r0 + weights * (r1 - r0)
        
        return r_flat.view(shape)
    
    

def launch(data_loader, actual_len, new_values, probabilities, traj_proc):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_trajectory_v_pre_15"

    if os.path.exists(f'./training/{args.run_name}'):
        pass
    else:
        os.makedirs(f'./training/{args.run_name}')

    args.in_c = actual_len
    # args.out_c = 5200

    args.emb_dim = 4
    # args.emb_dim = 164

    # args.epochs = 1500
    args.epochs = 341
    
    args.batch_size = 128
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # args.lr = 3e-4
    args.lr = 0.00005583
    
    args.save_file = f'./training/{args.run_name}/training_process_v_pre_1500_epochs.json'
    args.new_values = new_values
    args.probabilities = probabilities

    train(args, data_loader, actual_len, traj_proc)

def train(args, dataloader, actual_len, traj_proc):
    # for param in decoder.parameters():
    #     param.requires_grad = False
    
    device = args.device
    print(device)
    
    DifferentiableInverseQuantile_d = DifferentiableInverseQuantile(traj_proc.qt_d, output_distribution='normal').to(device)
    
    model = Pred_model(c_in=actual_len, c_out=actual_len, emb_dim=512, first_tim_dim=8, first_pos_dim=40, length_dim=128, device="cuda").to(device)
    # model = Net_model(c_in=actual_len, c_out=actual_len, emb_dim=512, first_tim_dim=8, first_pos_dim=40, length_dim=128, device="cuda").to(device)
    checkpoint = torch.load('./training/DDPM_trajectory_v_pre_15/ckpt_1159.pt')
    model.load_state_dict(checkpoint)

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
    

    start_epoch = 1160
    # start_epoch = 0
    
    for epoch in range(start_epoch, args.epochs + start_epoch):
        # print(f'starting epoch {epoch}:')
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
            combined, start_time_emb, emb_posi_seq, start_end_utm, mask_= batch
            
            combined = combined.to(device).float()
            start_time_emb = start_time_emb.to(device).float()
            mask_ = mask_.to(device)
            start_end_utm = start_end_utm.to(device).float()
            
            start_pos_emb = emb_posi_seq.to(device).float()  # 将 emb_posi_seq 转换为张量并移动到设备上
            
            # start_pos_emb = combined[:, :1, :128]  # 提取起始位置的编码
            # combined = combined[:, :, 128:]  # 去掉起始位置的编码，剩下的作为输入特征
            
            t = diffusion.sample_timesteps(combined.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(combined, t)
            # print(mask_[0, :20])
            
            # 同样确保 x_t 的填充位在进入模型前也是干净的
            # x_t = x_t * (~mask_).unsqueeze(-1)
            # print(f't after use in adding noise is on cuda: {t.is_cuda}')
            # predicted_noise = model(x_t, t)
            
            # start_flag_idx = torch.zeros_like(mask_, dtype=torch.int32).to(device)  # 创建与 mask_ 第一列相同形状的张量
            # start_flag_idx[:, 0] = 1
            # predicted_v = model(x_t, t, mask_, start_time_emb, start_pos_emb, start_flag_idx)
            
            predicted_v = model(x_t, t, mask_, start_time_emb, start_pos_emb)
            
            v_t = torch.sqrt(diffusion.alpha_hat[t][:, None, None])*noise - torch.sqrt(1-diffusion.alpha_hat[t][:, None, None])*combined                
            # 模型用来拟合v_t
            # loss = mse(v_t, predicted_v)
            
            # print(f'epoch: {epoch}, v_t.shape: {v_t.shape}, predicted_v.shape: {predicted_v.shape}')
            
            loss_ele_wise = F.mse_loss(v_t, predicted_v, reduction='none')
            
            weights = torch.ones_like(loss_ele_wise, requires_grad=False)
            
            weights[:, 0, :] = 0.0  # 将第0个点不参与损失计算
            # weights[:, :, :2] *= 10.0  # 给相对位置误差20倍权重
            # weights[:, :, 3] *= 5.0      # 给相对时间误差10倍权重
            weights = weights.to(device).float()
            
            # 只计算非 Padding 部分的 Loss 总和，并且乘以权重
            active_loss = (loss_ele_wise * weights * (~mask_).unsqueeze(-1)).sum()
            # print('active_loss type: ', active_loss.dtype)
            
            # 只除以有效元素的个数，而不是 batch_size * seq_len * feature_dim
            actual_num_elements = ((~mask_).sum() - mask_.size(0)) * loss_ele_wise.shape[-1]
            # print('actual_num_elements type: ', actual_num_elements.dtype)
            loss = active_loss / (actual_num_elements + 1e-8) # 防止除以0

            # x0_pred 是模型根据当前的 x_t 和预测的 v_t 反推回去的原始输入 x0 的估计值,代表对真实轨迹的差分表示, b * seq_l * 5
            # [d_transformed, np.sin(theta)*(1-is_stationary), np.cos(theta)*(1-is_stationary), dt_transformed, is_stationary]
            # 在表示时，轨迹第一个点的特征（5个维度）被强制设置为0，模型也不对第一个点进行预测，不使用第一个轨迹点的特征参数。
            x0_pred = torch.sqrt(diffusion.alpha_hat[t][:, None, None])*x_t - torch.sqrt(1-diffusion.alpha_hat[t][:, None, None])*predicted_v
            q_r_pred = x0_pred[:, :, 0]  # 模型预测的距离特征
            r_pred = DifferentiableInverseQuantile_d(q_r_pred)  # 将预测的距离特征通过可微的逆分位数变换得到物理距离
            
            # r_pred_ = traj_proc.qt_d.inverse_transform(q_r_pred.cpu().detach().numpy().reshape(-1, 1)).reshape(q_r_pred.shape[0], q_r_pred.shape[1])  # 将预测的距离特征通过可微的逆分位数变换得到物理距离
            # print('r_pred_ example: ', r_pred_[0, :])
            
            theta_pred = torch.atan2(x0_pred[:, :, 1], x0_pred[:, :, 2])
            sin_theta_pred = torch.sin(theta_pred)/torch.sqrt(torch.sin(theta_pred)**2 + torch.cos(theta_pred)**2 + 1e-8)
            cos_theta_pred = torch.cos(theta_pred)/torch.sqrt(torch.sin(theta_pred)**2 + torch.cos(theta_pred)**2 + 1e-8)
            dx_pred = r_pred * cos_theta_pred
            dy_pred = r_pred * sin_theta_pred
            
            # dx_pred = r_pred * torch.cos(theta_pred)
            # dy_pred = r_pred * torch.sin(theta_pred)
            
            dx_pred = dx_pred.masked_fill_(mask_, 0.0)  # 仅对非填充部分的dx进行计算
            dy_pred = dy_pred.masked_fill_(mask_, 0.0)  # 仅对非填充部分的dy进行计算
            
            dx_pred[:, 0] = 0.0  # 第0个点的dx为0
            dy_pred[:, 0] = 0.0  # 第0个点的dy为0

            
            total_dx = torch.cumsum(dx_pred, dim=1)[:, -1]  # 只取最后一个点的累计和，代表整个轨迹的总位移
            total_dy = torch.cumsum(dy_pred, dim=1)[:, -1]  # 只取最后一个点的累计和，代表整个轨迹的总位移
            
            # if cou == 601:
            #     print('combined example: ', combined[0, :20, :])
            #     print('r_pred_ example: ', r_pred_[0, :])
                
            #     print('r_pred example: ', r_pred[0, :])
                
            #     print('dx_pred example: ', dx_pred[0, :])
            #     print('dy_pred example: ', dy_pred[0, :])
            #     print('total_dx example: ', total_dx)
            #     print('total_dy example: ', total_dy)
                
            #     print('start_x: ', start_end_utm[:, 0, 0])
            #     print('start_y: ', start_end_utm[:, 0, 1])
                
            #     print('end_x: ', start_end_utm[:, 1, 0])
            #     print('end_y: ', start_end_utm[:, 1, 1])
            
            end_x_pred = start_end_utm[:, 0, 0] + total_dx
            end_y_pred = start_end_utm[:, 0, 1] + total_dy
            
            end_xy = torch.stack([end_x_pred, end_y_pred], dim=-1)
            
            # loss_end = F.mse_loss(end_xy, start_end_utm[:, 1, :], reduction='mean')
            
            raw_endpoint_loss_per_sample = F.huber_loss(end_xy, start_end_utm[:, 1, :], delta=50.0, reduction='none').mean(dim=-1)
        
            # 应用基于扩散时间步 t 的信噪比权重 (alpha_bar_t)
            alpha_bar_weight = diffusion.alpha_hat[t]  # [B]
            
            t_weighted_endpoint_loss = raw_endpoint_loss_per_sample * alpha_bar_weight
            
            # 计算最终的 Batch Mean Endpoint Loss
            final_endpoint_loss = t_weighted_endpoint_loss.mean()
            
            # 应用常数缩放 与 Epoch Warm-up
            if epoch < 100:
                epoch_weight = 0.0
            else:
                # epoch_weight = (epoch - 100) / (args.epochs - 100)
                epoch_weight = (epoch - 100) / (1500 - 100)
                
            # epoch_weight = torch.tensor(epoch_weight, device=device, dtype=torch.float32)
            
            total_loss = loss + 0.001 * epoch_weight * final_endpoint_loss
            
            
            if cou == 601:
                print('loss_end: ', final_endpoint_loss.item())
                
            # 忽略 Padding 部分的预测结果，计算有效部分的物理距离和角度误差
            # loss = (loss_ele_wise * weights * (~batch_mask).unsqueeze(-1)).mean()
            # total_loss = loss + 5e-8*loss_end
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # x0 = torch.cat([m_feat 128, t_feat 128, f_feat 256], dim=-1)
            if cou % 600 == 0 and epoch % 50 == 0:  # 每 100 个 batch 检查一次
                check_input_layer_gradients(model, flag_dim=128, physics_dim=128)
            
            optimizer.step()

            copy_loss = total_loss.clone()
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
    
    f = open('./filtered_trips_posi_seq.json', 'r')
    trips_posi_seq = json.load(f)
    f.close()

    f = open('./filtered_trips_time_seq.json', 'r')
    trips_time_seq = json.load(f)
    f.close()
    
    seq_l = 120
    # max_T = 20000
    # max_T = 30587
    # emb_posi_seq, emb_posi_seq_d, emb_time_seq, abs_emb_time_seq, mask_ = construct_features(time_seq[:10], posi_seq[:10], 520)
    # emb_posi_seq, emb_posi_seq_d, emb_time_seq, abs_emb_time, abs_emb_time_seq, start_time_emb, mask_ = construct_features(v_thre_time_seq, v_thre_posi_seq, 216)
    # emb_posi_seq, emb_posi_seq_d, emb_time_seq, start_time_emb, mask_ = construct_features(trips_time_seq, trips_posi_seq, seq_l)
    emb_posi_seq, emb_posi_seq_d, emb_time_seq, start_time_emb, start_end_utm, mask_ = construct_features(trips_time_seq, trips_posi_seq, seq_l)
    
    print(start_time_emb.dtype)
    print(start_end_utm.shape)
    
    all_dx = []
    all_dy = []
    all_dt = []

    trace_count = 0
    for i in range(mask_.shape[0]):
        actual_len = len(mask_[i]) - mask_[i].sum().item()
        
        ds_dt_count = 0
        # 不考虑第一个点，因为第一个点是轨迹的起点被设置为全0，没有意义
        for j in range(1,actual_len):
            x = emb_posi_seq_d[i, j, 0]
            y = emb_posi_seq_d[i, j, 1]
            all_dx.append(x)
            all_dy.append(y)
            all_dt.append(emb_time_seq[i][j, 0])
    
    
    traj_proc = TrajectoryProcessor(eps=0.5)
    # fit的时候要拉成一个一维长向量
    # transforme的时候要保持原来的形状，最后输出的feature也是和输入形状一样的
    traj_proc.fit(np.array(all_dx), np.array(all_dy), np.array(all_dt))
    
    # 变换中对极坐标表示的r,时间差dt采用了quantile transform，这些是参与扩散的变量
    feat = traj_proc.transformer(emb_posi_seq_d[:, :, 0], emb_posi_seq_d[:, :, 1], np.array(emb_time_seq)[:, :, 0])
    print('feat shape: ', feat.shape)
    feat[mask_] = 0.0  # 将填充部分的特征设置为0，确保它们在训练中不产生影响
    print(feat[23, :32, :])
    
    joblib.dump(traj_proc, "d_t_quantile_scaler.pkl")
    
    lengths = np.array([len(seq)-sum(seq) for seq in mask_])
    print(lengths[:100])
    
    sorted_indices = np.argsort(lengths)
    # np.argsort返回的是下标，即把lengths数组从小到大排序后对应的原始下标。sorted_indices[0]是最短轨迹的下标，sorted_indices[-1]是最长轨迹的下标。
    lengths_sorted = lengths[sorted_indices]
    print('lengths_sorted: ', lengths_sorted[:10])
    
    print(sorted_indices[0])
    # print(norm_emb_time_seq[sorted_indices[0]])
    
    # 去重后95%分位数对应的轨迹长度比不去重的更长，实际上重复长度的轨迹应该计算数量。去重后0.95分位数对应的轨迹长度是10349，而不去重的0.95分位数对应的轨迹长度是5204。
    i = int(len(lengths)*0.95)+1
    print(lengths_sorted[i])
    
    values, counts = np.unique(lengths_sorted, return_counts=True)
    # plt.bar(values, counts)
    
    new_values = []
    new_counts = []
    
    for value, count in zip(values, counts):
        if value <= seq_l:
            new_values.append(value)
            new_counts.append(count)
    new_values = np.array(new_values)
    new_counts = np.array(new_counts)
    print('len(new_values) ', len(new_values))
    # plt.plot(new_values, new_counts)
    # plt.show()
    
    probabilities = new_counts / new_counts.sum()
    print('sum_probability ', probabilities.sum())
    
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
    
    # # 创建 [L, 1] 的 flag，第一点为 1，其他点为 0
    # start_flag = np.zeros((b, seq_l, 1), dtype=np.float32)
    # start_flag[:, 0, 0] = 1.0  # 仅第一点为 1
    
    # # 拼接后特征维度变为 D + 1
    # combined = np.concatenate([start_flag, combined], axis=-1)
    # print('combined.shape: ', combined.shape)
    

    zipped_dataset = ZipDataset(combined, start_time_emb, emb_posi_seq, start_end_utm, mask_)
    dataloader = DataLoader(zipped_dataset, batch_size=128, shuffle=True, drop_last=False)
    
    # sta_dict = torch.load('./coord_decoder.pt')
    # decoder = CoordinateDecoder(input_dim=128, hidden_dim=512, num_blocks=2)
    # decoder.load_state_dict(state_dict=sta_dict)
    
    launch(dataloader, seq_l, new_values, probabilities, traj_proc)
    