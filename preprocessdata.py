from modules import *


json_file = './all_trajectories.json'
time_seq, posi_seq = get_geolife_data(json_file)
print('len(time_seq), len(posi_seq): ', len(time_seq), len(posi_seq))
print('len(time_seq[0]), len(posi_seq[0]): ', len(time_seq[0]), len(posi_seq[0]))

new_time_seq = []
new_posi_seq = []

time_len_record = []
posi_len_record = []

for i in range(len(time_seq)):
    one_time_seq = time_seq[i]
    one_posi_seq = posi_seq[i]
    
    new_one_time_seq = []
    new_one_posi_seq = []
    
    for j in range(1, len(one_time_seq)):
        pre_time = one_time_seq[j-1]
        pre_lon, pre_lat = one_posi_seq[j-1]
        
        time = one_time_seq[j]
        lon, lat = one_posi_seq[j]
        
        new_one_time_seq.append(pre_time)
        new_one_posi_seq.append(one_posi_seq[j-1])
        
        if (time-pre_time) > 600.0:
            new_posi_seq.append(new_one_posi_seq)
            new_time_seq.append(new_one_time_seq)
            time_len_record.append(len(new_one_time_seq))
            posi_len_record.append(len(new_one_posi_seq))
            
            new_one_time_seq = []
            new_one_posi_seq = []

print('len(new_time_seq), len(new_posi_seq): ', len(new_time_seq), len(new_posi_seq))
print('len of time_len_record: ', len(time_len_record))
print('average len and min max len of trails: ', sum(posi_len_record)/len(posi_len_record), max(posi_len_record), min(posi_len_record))
print('new_time_seq[0][0]：', new_time_seq[0][0])


v_thre_time_seq = []
v_thre_posi_seq = []

inverse_time_count = 0

for i in range(len(new_time_seq)):
    # v_thre_one_time_seq = []
    # v_thre_one_posi_seq = []
    count_abnormal = 0
    
    for j in range(1, len(new_time_seq[i])):
        pre_time = new_time_seq[i][j-1]
        pre_lon, pre_lat = new_posi_seq[i][j-1]
        
        time = new_time_seq[i][j]
        lon, lat = new_posi_seq[i][j]
        
        d_lon = lon - pre_lon
        d_lat = lat - pre_lat
        # ave_lat = (lat + pre_lat) / 2
        ave_lat = pre_lat
        
        lat_ave_rad = math.radians(ave_lat)
        d_lon_rad = math.radians(d_lon)
        d_lat_rad = math.radians(d_lat)
        
        dx = R * d_lon_rad * math.cos(lat_ave_rad)
        dy = R * d_lat_rad
        dt = time - pre_time
        if dt < 0:
            inverse_time_count += 1
        
        v = math.sqrt(dx*dx + dy*dy) / (dt+1e-6)
        
        # if v > 200.0:
        #     print(f'异常点： {lon}, {lat}, {time}, {pre_lon}, {pre_lat}, {pre_time}, dx: {dx}, dy: {dy}, 速度：{v}, 时间: {dt}')
            
        if v > 45.0:
            # print(f'异常点： {lon}, {lat}, {time}, {pre_lon}, {pre_lat}, {pre_time}, dx: {dx}, dy: {dy}, 速度：{v}')
            count_abnormal += 1
            break

            
    if count_abnormal < 1:
        v_thre_time_seq.append(new_time_seq[i])
        v_thre_posi_seq.append(new_posi_seq[i])

print('len(v_thre_time_seq), len(v_thre_posi_seq): ', len(v_thre_time_seq), len(v_thre_posi_seq))

print('inverse_time_count: ', inverse_time_count)

f = open('v_thre_time_seq_new.json', 'w')
json.dump(v_thre_time_seq, f, ensure_ascii=False)
f.close()

f = open('v_thre_posi_seq_new.json', 'w')
json.dump(v_thre_posi_seq, f, ensure_ascii=False)
f.close()
