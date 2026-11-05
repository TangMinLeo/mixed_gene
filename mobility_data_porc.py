import pandas as pd
import json
import os

# data_file = r'D:\data\ods_bd_grid_202512191120.csv'
# df = pd.read_csv(data_file, encoding='utf-8')
# print(df.head())

# print(df.columns.tolist())

# selected_columns = ['netID', 'wkt']

# selected_df = df[selected_columns]

# grid_dict = {}
# ff = open(f'./dict_grid.json', 'w')

# for i in range(len(selected_df)):
#     if i < 10:
#         # print(selected_df.iloc[i])
#         # print(type(selected_df.iloc[i]['wkt']))
#         wkt = selected_df.iloc[i]['wkt']
#         sep = wkt.split('((')[1].split('))')[0].split(', ')
#         print(sep)
#     # grid_dict[selected_df.iloc[i]['netID']] = selected_df.iloc[i]['wkt']
    
#     sep = wkt.split('((')[1].split('))')[0].split(', ')
#     coords = []
#     for point in sep:
#         lon, lat = point.split(' ')
#         coords.append((float(lon), float(lat)))
#     grid_dict[selected_df.iloc[i]['netID']] = coords
    
# print(len(grid_dict))
# json.dump(grid_dict, ff, ensure_ascii=False)
# ff.close()

dir_path = r'D:\BaiduNetdiskDownload\轨迹数据20250616\ods_bd_person_trajectory'
files = os.listdir(dir_path)


df_traces = pd.read_csv(os.path.join(dir_path, files[0]), encoding='utf-8')

for jj in range(1, len(files)):
    print(files[jj])
    temp = pd.read_csv(os.path.join(dir_path, files[jj]), encoding='utf-8')
    
    df_traces = pd.concat([df_traces, temp], ignore_index=True)


# traces_file = r'D:\BaiduNetdiskDownload\轨迹数据20250616\ods_bd_person_trajectory\part-00000-d8808df1-c7bf-42af-a0ca-3ebdcd2c6f49-c000.csv'
# df_traces = pd.read_csv(traces_file, encoding='utf-8') 


print(len(df_traces))

traces_dict_by_person = {}

persons = df_traces['uuid'].unique().tolist()
print(f'总用户数: {len(persons)}')

for person in persons:
    traces_dict_by_person[person] = []

# start_time = df_traces['start_time'].astype(int).to_list()
# date_str = df_traces['date_str'].unique.to_list()
# net_id = df_traces['netid_100'].to_list()
# uuid = df_traces['uuid'].to_list()
# stay_minutes = df_traces['stay_minutes'].to_list()


only_every_day_traces = []
new_only_every_day_traces = []
count = 0
for person in persons:
    count += 1
    # if count < 5:
    one_person = df_traces[df_traces['uuid'] == person]
    # print(one_person.head())
    
    date_str = one_person['date_str'].unique().tolist()
    # print(date_str)
    
    for date_ in date_str:
    
        one_day = one_person[one_person['date_str'] == date_]
        sorted_one_day = one_day.sort_values(by='start_time')
        # print(type(sorted_one_day.iloc[0]['start_time']))
        
        one_day_list = []
        one_day_traces = []
        new_one_day_list = []
        for i in range(len(sorted_one_day)):
            trace_point = {
            'start_time': int(sorted_one_day.iloc[i]['start_time']),
            'netid_100': sorted_one_day.iloc[i]['netid_100'],
            'stay_minutes': int(sorted_one_day.iloc[i]['stay_minutes']),
            'date_str': sorted_one_day.iloc[i]['date_str']
            }
            
            long_lat = (sorted_one_day.iloc[i]['lnt_wgs84'], sorted_one_day.iloc[i]['lat_wgs84'], int(sorted_one_day.iloc[i]['start_time']))
            new_one_day_list.append(long_lat)
            
            one_day_list.append(trace_point)
            
            one_day_traces.append((sorted_one_day.iloc[i]['netid_100'], int(sorted_one_day.iloc[i]['start_time']), int(sorted_one_day.iloc[i]['stay_minutes'])))
        
        traces_dict_by_person[person].append(one_day_list)
        only_every_day_traces.append(one_day_traces)
        # print(len(one_day_traces))
        
        new_only_every_day_traces.append(new_one_day_list)
        
f_forshow = open(f'./day_traces_1.json', 'w')
json.dump(new_only_every_day_traces, f_forshow, ensure_ascii=False)
f_forshow.close()
print(f'finised, total persion count: {count}')
        
    # if count < 5000000:
    #     pass
    #     # print(person)
    #     # print(traces_dict_by_person[person])
    # else:
    #     f_forshow = open(f'./day_traces_1.json', 'w')
    #     json.dump(new_only_every_day_traces, f_forshow, ensure_ascii=False)
    #     f_forshow.close()
    #     break

# ff_traces = open(f'./dict_traces_by_person.json', 'w')
# json.dump(traces_dict_by_person, ff_traces, ensure_ascii=False)
# ff_traces.close()

# ff_only_day = open(f'./anonyed_every_day_traces.json', 'w')
# json.dump(only_every_day_traces, ff_only_day, ensure_ascii=False)
# ff_only_day.close()

# print(len(only_every_day_traces))

