import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

file_path = './training/DDPM_trajectory_v_pre_1/training_process_v_pre_1500_epochs.json'
# file_path = './training_process.json'

with open(file_path, 'r') as f:
    data = json.load(f)

    values = []
    min_ave = 100000.0
    idx = 0
    for key in data.keys():
        values.append(data[key][0])
        if float(data[key][0]) < min_ave:
            min_ave = float(data[key][0])
            idx = int(key)

    print(min_ave)
    print(idx)
    # plt.plot(values[100:])
    plt.plot(values)
    plt.show()


# csv_file = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'
# dict_dims = {}
# idx_map = {}
#
# with open(csv_file, 'r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     count = 0
#     for row in reader:
#         if count == 0:
#             col_count = 0
#             for item in row:
#                 dict_dims[item] = []
#                 idx_map[col_count] = item
#                 col_count += 1
#             count = count + 1
#         else:
#             for j in range(len(row)):
#                 dict_dims[idx_map[j]].append(row[j])
#             count = count + 1
#
#     print(count)
# # print(dict_dims)
#
# unique_dims = {}
# total_ans = 0
# for key in dict_dims.keys():
#     # print(key, len(dict_dims[key]))
#     unique_dims[key] = list(set(dict_dims[key]))
#     total_ans = total_ans + len(unique_dims[key])
#     # print(unique_dims[key])
#
# print(total_ans)
#
# count_dict_total = {}
# for key in dict_dims.keys():
#     count_dict_total[key] = {}
#     for item in unique_dims[key]:
#         count_dict_total[key][item] = 0
#
#     total_num = 0
#     for ans in dict_dims[key]:
#         count_dict_total[key][ans] = count_dict_total[key][ans] + 1
#         total_num = total_num + 1
#     print(total_num)
#     print(count_dict_total[key])
#
#     for item in unique_dims[key]:
#         count_dict_total[key][item] = count_dict_total[key][item]/total_num
#     print(count_dict_total[key])


# code_map = {}
#
# code_map['Age:[5,10)'] = '[5,10)'
# code_map['Age:[10,15)'] = '[10,15)'
# code_map['Age:[15,20)'] = '[15,20)'
# code_map['Age:[20,25)'] = '[20,25)'
# code_map['Age:[25,30)'] = '[25,30)'
# code_map['Age:[30,35)'] = '[30,35)'
# code_map['Age:[35,40)'] = '[35,40)'
# code_map['Age:[40,45)'] = '[40,45)'
# code_map['Age:[45,50)'] = '[45,50)'
# code_map['Age:[50,55)'] = '[50,55)'
# code_map['Age:[55,60)'] = '[55,60)'
# code_map['Age:[60,65)'] = '[60,65)'
# code_map['Age:[65,70)'] = '[65,70)'
# code_map['Age:[70,75)'] = '[70,75)'
# code_map['Age:[75,80)'] = '[75,80)'
# code_map['Age:[80,85)'] = '[80,85)'
# code_map['Age:[85,90]'] = '[85,90]'
#
# code_map['Gender:1'] = 'male'
# code_map['Gender:2'] = 'female'
#
# code_map['Homeincome:1'] = '<1 million'
# code_map['Homeincome:2'] = '1 million-3 million'
# code_map['Homeincome:3'] = '3 million-5 million'
# code_map['Homeincome:4'] = '5 million-10 million'
# code_map['Homeincome:5'] = '>10 million'
#
# code_map['Hometype:1'] = 'apartment'
# code_map['Hometype:2'] = 'villa'
# code_map['Hometype:3'] = 'multi-family'
# code_map['Hometype:4'] = 'single house'
# code_map['Hometype:5'] = 'dual purpose house'
# code_map['Hometype:6'] = 'other'
#
# code_map['CarOwn:1'] = 'yes'
# code_map['CarOwn:2'] = 'no'
#
# code_map['Driver:1'] = 'yes'
# code_map['Driver:2'] = 'no'
#
# code_map['Workdays:1'] = '5 days per week'
# code_map['Workdays:2'] = '6 days per week'
# code_map['Workdays:3'] = '1-4 days per week'
# code_map['Workdays:4'] = 'inoccupation/non-regular'
#
# code_map['Worktype:1'] = 'student'
# code_map['Worktype:2'] = 'inoccupation/housewife'
# code_map['Worktype:3'] = 'experts'
# code_map['Worktype:4'] = 'service'
# code_map['Worktype:5'] = 'sales'
# code_map['Worktype:6'] = 'manager/office'
# code_map['Worktype:7'] = 'agriculture and fisher'
# code_map['Worktype:8'] = 'simple labor'
# code_map['Worktype:9'] = 'others'
#
# code_map['Student:1'] = 'elementary/middle/high'
# code_map['Student:2'] = 'kid'
# code_map['Student:3'] = 'university'
# code_map['Student:4'] = 'none'
#
# code_map['NumHH:1'] = '1'
# code_map['NumHH:2'] = '2'
# code_map['NumHH:3'] = '3'
# code_map['NumHH:4'] = '4'
# code_map['NumHH:5'] = '5'
# code_map['NumHH:6'] = '6'
# code_map['NumHH:7'] = '7'
#
# code_map['KidinHH:1'] = 'yes'
# code_map['KidinHH:2'] = 'no'
#
# code_map['ComMode:Walking'] = 'Walking'
# code_map['ComMode:Car'] = 'Car'
# code_map['ComMode:PM'] = 'bike'
# code_map['ComMode:None'] = 'None'
# code_map['ComMode:Taxi'] = 'Taxi'
# code_map['ComMode:PT'] = 'public transportation'
#
# code_map['ComTime:Peak'] = 'Peak'
# code_map['ComTime:NonPeak'] = 'NonPeak'
# code_map['ComTime:Others'] = 'Others'
# code_map['ComTime:None'] = 'None'
