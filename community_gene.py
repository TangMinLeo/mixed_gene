# from copula_tools import *
import json
import pandas as pd
from modules import Net_model, Diffusion
import torch
from tqdm import tqdm
import numpy as np
from modules import get_geolife_data
from decode_traj import CoordinateDecoder
import math


min_lon = 115.416667
max_lon = 117.500000
min_lat = 39.433333
max_lat = 41.050000

mea_delta = 3.1487522331724422
std_delta = 0.9709251038940904
# ff = open('./copula_popu_gene_acs_1_10000.json', 'w', encoding='utf-8')
# json.dump(all_row_decode, ff, ensure_ascii=False)
# ff.close()


if __name__ == '__main__':
    print('begin generating...')
    save_json_file = './1000_tempspa_traj_gene_0.json'
    model = Net_model(c_in=520, c_out=520, emb_dim=512, device="cuda").to(torch.device('cuda'))
    checkpoint = torch.load('./training/DDPM_trajectory_v_pre_3/ckpt_1498_.pt')
    model.load_state_dict(checkpoint)
    
    json_file = './all_trajectories.json'
    time_seq, posi_seq = get_geolife_data(json_file)
    print('len(time_seq), len(posi_seq): ', len(time_seq), len(posi_seq))
    
    lengths = np.array([len(seq) for seq in time_seq])
    sorted_indices = np.argsort(lengths)
    # np.argsort返回的是下标，即把lengths数组从小到大排序后对应的原始下标。sorted_indices[0]是最短轨迹的下标，sorted_indices[-1]是最长轨迹的下标。
    lengths_sorted = lengths[sorted_indices]
    print('lengths_sorted: ', lengths_sorted[:10])
    
    print(sorted_indices[0])
    print(time_seq[sorted_indices[0]])
    
    # 去重后95%分位数对应的轨迹长度比不去重的更长，实际上重复长度的轨迹应该计算数量。去重后0.95分位数对应的轨迹长度是10349，而不去重的0.95分位数对应的轨迹长度是5204。
    i = int(len(lengths)*0.95)+1
    print(lengths_sorted[i])
    
    values, counts = np.unique(lengths_sorted, return_counts=True)
    # plt.bar(values, counts)
    
    new_values = []
    new_counts = []
    
    for value, count in zip(values, counts):
        if value <= 520:
            new_values.append(value)
            new_counts.append(count)
    new_values = np.array(new_values)
    new_counts = np.array(new_counts)
    
    # plt.plot(new_values, new_counts)
    # plt.show()
    
    probabilities = new_counts / new_counts.sum()

    model.eval()
    diffusion = Diffusion(noise_steps=2000, beta_start=1e-4, beta_end=0.02, channels=520, emb_dim=136, device=torch.device('cuda'))

    # q = 1000//300
    # r = 1000 % 300
    # all_sample = np.array([])
    # all_sample_mask = []
    # for _ in range(q):
        # sampled_items = diffusion.sample(model, 300, new_values, probabilities)
        # sampled_items, sampled_mask = diffusion.sample(model, 300, new_values, probabilities)
        # all_sample.extend(sampled_items.tolist())
        # all_sample_mask.extend(sampled_mask.tolist())

    # sampled_items, sampled_mask = diffusion.sample(model, r, new_values, probabilities)
    # all_sample.extend(sampled_items.tolist())
    # all_sample_mask.extend(sampled_mask.tolist())

    sampled_items, sampled_mask = diffusion.sample(model, 100, new_values, probabilities)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_freqs = 32
    input_dim = num_freqs * 4
    
    decoder = CoordinateDecoder(input_dim=input_dim, hidden_dim=512, num_blocks=2).to(device)
    sta_dic = torch.load('./coord_decoder.pt')
    decoder.load_state_dict(state_dict=sta_dic)
    
    decoder.eval()
    for i in range(len(sampled_items)):
        one_traj = sampled_items[i,:,:]
        traj_len = np.sum(~sampled_mask[i,:])
        
        traj_real = one_traj[:traj_len, :]
        one_batch = torch.from_numpy(traj_real).to(device)
        
        with torch.no_grad():
            decoded = decoder(one_batch[:,:128])
            time = one_batch[:,128:]
            dec = decoded.detach().cpu().numpy()
            for i in range(len(dec)):
                poit = dec[i,:]
                print(f'{(poit[0]+1)*(max_lon-min_lon)/2+min_lon}, {(poit[1]+1)*(max_lat-min_lat)/2+min_lat},')
                
            for i in range(len(dec)):
                one_t = time[i,:].cpu().numpy()
                
                rad_m = math.atan2(one_t[1], one_t[2])
                rad_m_norm = rad_m % (2 * math.pi)
                m_final = int(rad_m_norm / (2 * math.pi) * 60)
                
                rad_h = math.atan2(one_t[3], one_t[4])
                rad_h_norm = rad_h % (2 * math.pi)
                h_final = int(rad_h_norm / (2 * math.pi) * 24)
                
                rad_d = math.atan2(one_t[5], one_t[6])
                rad_d_norm = rad_d % (2 * math.pi)
                d_final = int(rad_d_norm / (2 * math.pi) * 7)
                
                second_offset = np.exp(one_t[7] * std_delta + mea_delta)

                seconds = int(second_offset)
                print(f'day_of_week: {d_final}, {h_final}:{m_final}, {seconds}')
                
            print('one trajectory is decoded!!!')
        
        
        
        