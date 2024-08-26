import pandas as pd
import numpy as np
import os

def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    in_dict_info = False
    in_matrix_bus = False
    in_matrix_branch = False
    in_matrix_gen = False

    dict_info = {}
    matrix_bus_data = []
    matrix_branch_data = []
    matrix_gen_data = []

    matrix_bus_columns = []
    matrix_branch_columns = []
    matrix_gen_columns = []

    for line in lines:
        line = line.strip()

        if line == "#####Begin Matrix Bus#####":
            in_matrix_bus = True
            continue
        elif line == "#####End Matrix Bus#####":
            in_matrix_bus = False
            continue
        elif line == "#####Begin Matrix Branch#####":
            in_matrix_branch = True
            continue
        elif line == "#####End Matrix Branch#####":
            in_matrix_branch = False
            continue
        elif line == "#####Begin Matrix Gen#####":
            in_matrix_gen = True
            continue
        elif line == "#####End Matrix Gen#####":
            in_matrix_gen = False
            continue

        if in_matrix_bus:
            if not matrix_bus_columns: # 因为一开始是空的，所以肯定先运行这一步。修改后就走下面的流程了
                # The first line in the Matrix section is the header
                matrix_bus_columns = line.split(',')
            else:
                # The following lines are data
                matrix_bus_data.append(list(map(float, line.split(','))))
        
        if in_matrix_branch:
            if not matrix_branch_columns: # 因为一开始是空的，所以肯定先运行这一步。修改后就走下面的流程了
                # The first line in the Matrix section is the header
                matrix_branch_columns = line.split(',')
            else:
                # The following lines are data
                matrix_branch_data.append(list(map(float, line.split(','))))
        
        if in_matrix_gen:
            if not matrix_gen_columns:
                matrix_gen_columns = line.split(',')
            else:
                matrix_gen_data.append(list(map(float, line.split(','))))

    # Convert the matrix data to a DataFrame
    matrix_bus_df = pd.DataFrame(matrix_bus_data, columns=matrix_bus_columns)
    matrix_branch_df = pd.DataFrame(matrix_branch_data, columns=matrix_branch_columns)
    matrix_gen_df = pd.DataFrame(matrix_gen_data, columns=matrix_gen_columns)
    for i in range(len(matrix_gen_df)):
        index = matrix_gen_df["GEN_BUS"][i]
        # print(index)
        matrix_bus_df['PD'][index-1] = matrix_bus_df['PD'][index-1] - matrix_gen_df["PG"][i]
        # print(matrix_bus_df['PD'][index-1])
        matrix_bus_df['QD'][index-1] = matrix_bus_df['QD'][index-1] - matrix_gen_df["QG"][i]
        # print(matrix_bus_df['QD'][index-1])

    matrix_bus_df = matrix_bus_df[['BUS_I','BUS_TYPE','VM','VA','PD','QD']]
    matrix_bus_df['PD'], matrix_bus_df['QD'] = matrix_bus_df['PD']/100, matrix_bus_df['QD']/100
    matrix_bus_df['BUS_TYPE'] = matrix_bus_df['BUS_TYPE'].replace({3: 0, 1: 2, 2: 1}) # 和给定的bus type统一
    matrix_bus_df['BUS_I'] = matrix_bus_df['BUS_I'] - 1 # index数值改成从0开始
    matrix_branch_df = matrix_branch_df[['F_BUS', 'T_BUS','BR_R','BR_X' ]]
    matrix_branch_df['F_BUS'], matrix_branch_df['T_BUS'] = matrix_branch_df['F_BUS']-1, matrix_branch_df['T_BUS']-1
    return  matrix_bus_df,matrix_branch_df



case_nums = 500
store_matrix_bus = []
store_matrix_branch = []

file_folder_path = '/mnt/afs/data/Samples_v5/case118/caseN/'
file_prename = 'IEEE118,case'
file_endname = ',FL0.txt'

for i in range(case_nums):
    filename = file_prename + str(i) + file_endname
    file_path = os.path.join(file_folder_path,filename)
    matrix_bus_df, matrix_branch_df = parse_txt_file(file_path)


    if not store_matrix_bus:
        store_matrix_bus.append(matrix_bus_df.values.tolist())
    else:
        store_matrix_bus.append(matrix_bus_df.values.tolist())
    if not store_matrix_branch:
        store_matrix_branch.append(matrix_branch_df.values.tolist())
    else:
        store_matrix_branch.append(matrix_branch_df.values.tolist())
    
# case_nums = 29500
case_nums = 100000

# store_matrix_bus = []
# store_matrix_branch = []

file_folder_path = '/mnt/afs/data/Samples_v5/case118/caseN_1/'
file_prename = 'IEEE118,case'
file_endname = ',FL1.txt'

for i in range(case_nums):
    filename = file_prename + str(i) + file_endname
    file_path = os.path.join(file_folder_path,filename)
    matrix_bus_df, matrix_branch_df = parse_txt_file(file_path)


    if not store_matrix_bus:
        store_matrix_bus.append(matrix_bus_df.values.tolist())
    else:
        store_matrix_bus.append(matrix_bus_df.values.tolist())
    if not store_matrix_branch:
        store_matrix_branch.append(matrix_branch_df.values.tolist())
    else:
        store_matrix_branch.append(matrix_branch_df.values.tolist())

np.save('./case118n01_vlarge_node_features.npy', np.array(store_matrix_bus)) # 这里也记得改一下
np.save('./case118n01_vlarge_edge_features.npy',np.array(store_matrix_branch))


# np.save('./case118n01_vsmall_node_features.npy', np.array(store_matrix_bus)) # 这里也记得改一下
# np.save('./case118n01_vsmall_edge_features.npy',np.array(store_matrix_branch))

print("保存的 store_matrix_bus 数组形状:", np.array(store_matrix_bus).shape)
print("保存的 store_matrix_branch 数组形状:", np.array(store_matrix_branch).shape)
# print(np.array(store_matrix_bus))