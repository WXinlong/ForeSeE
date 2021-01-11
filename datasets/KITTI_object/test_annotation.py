import json
import os, sys

# MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.append(os.path.join(MY_DIRNAME))

def data_object_training_annotations():
    file_path = "annotations/data_object_training_annotations.json"
    with open(file_path) as f:
        a = json.load(f)
    print(len(a))
    print(a[0])
    print(len(a[0]))

def inference_annotations():
    file_path = "annotations/inference_annotations.json"
    with open(file_path) as f:
        a = json.load(f)
    print(len(a))
    print(a[0])
    print(a[1])

def object_inference_annotations():
    file_path = "annotations/object_inference_annotations.json"
    with open(file_path) as f:
        a = json.load(f)
    print(len(a))
    print(a[0])

def train_annotations():
    file_path = "annotations/train_annotations.json"
    with open(file_path) as f:
        a = json.load(f)
    print(len(a))
    print(a[135])
    print(len(a[135]))
    
    # print(len(a))
    # print(a[3621]['rgb_path'][17:-4])
    f = open('train.txt', 'r')
    content = f.read()
    f.close()
    list_train = []
    for i in range(3712):
        # print(type(content[i*7:(i+1)*7]))
        # print(content[i*7:(i+1)*7])
        # print(len(content[i*7:(i+1)*7]))
        
        list_train.append(content[i*7:(i+1)*7][:6])
    # print(len(list_train))
    
    for i in range(3622):
        # if a[i]['rgb_path'][17:-4] in list_train:
        #     print("hihi")
        # print(a[i]['rgb_path'][17:-4])
        # print(type(a[i]['rgb_path'][17:-4]))
        # print("###")
        
        list_train.remove(a[i]['rgb_path'][17:-4])

    
        

    print(list_train)


    # print(a[0])
    # print(a[1])
    # for i in a:
    #     print(i['rgb_path'])

    # print(len(a))
    # print(a[1])
    # print(a[1]['depth_path'])
    

def val_annotations():
    file_path = "annotations/val_annotations.json"
    with open(file_path) as f:
        a = json.load(f)
    print(len(a))
    print(a[0])
    print(len(a[0]))
    
    f = open('val.txt', 'r')
    content = f.read()
    f.close()
    list_val = []
    for i in range(3769):
        # print(content[i*7:(i+1)*7])
        list_val.append(content[i*7:(i+1)*7][:6])
    # print(len(list_val))

    for i in range(3696):
        # if a[i]['rgb_path'][17:-4] in list_train:
        #     print("hihi")
        # print(a[i]['rgb_path'][17:-4])
        # print(type(a[i]['rgb_path'][17:-4]))
        # print("###")
        list_val.remove(a[i]['rgb_path'][17:-4])

    print(len(list_val))
    print(list_val)
    


if __name__ == "__main__":
    # data_object_training_annotations()
    # inference_annotations()
    # object_inference_annotations()
    # train_annotations()
    val_annotations()
    
    # train_miss = ['000013', '000217', '000256', '000295', '000339', '000421', '000462', '000501', '000623', '000637', '000673', '000947', '001091', '001165', '001248', '001282', '001396', '001444', '001509', '001556', '001584', '001611', '001700', '001761', '001806', '001838', '001947', '002199', '002247', '002335', '002352', '002364', '002443', '002667', '002739', '002774', '003077', '003184', '003307', '003589', '003638', '003696', '003858', '003921', '003942', '004030', '004039', '004058', '004070', '004204', '004225', '004389', '004427', '004580', '004645', '004755', '004818', '004854', '004991', '005084', '005301', '005387', '005392', '005393', '005421', '005504', '005761', '005808', '005898', '005920', '006021', '006072', '006079', '006203', '006485', '006573', '006609', '006610', '006661', '006740', '006912', '006920', '006979', '007018', '007025', '007092', '007143', '007293', '007338', '007390']
    # 000013.png 000217.png 000256.png 000295.png 000339.png 000421.png 000462.png 000501.png 000623.png 000637.png 000673.png 000947.png 001091.png 001165.png 001248.png 001282.png 001396.png 001444.png 001509.png 001556.png 001584.png 001611.png 001700.png 001761.png 001806.png 001838.png 001947.png 002199.png 002247.png 002335.png 002352.png 002364.png 002443.png 002667.png 002739.png 002774.png 003077.png 003184.png 003307.png 003589.png 003638.png 003696.png 003858.png 003921.png 003942.png 004030.png 004039.png 004058.png 004070.png 004204.png 004225.png 004389.png 004427.png 004580.png 004645.png 004755.png 004818.png 004854.png 004991.png 005084.png 005301.png 005387.png 005392.png 005393.png 005421.png 005504.png 005761.png 005808.png 005898.png 005920.png 006021.png 006072.png 006079.png 006203.png 006485.png 006573.png 006609.png 006610.png 006661.png 006740.png 006912.png 006920.png 006979.png 007018.png 007025.png 007092.png 007143.png 007293.png 007338.png 007390.png
    # print(len(train_miss))

    # train_miss = [i +'.png' for i in train_miss] 
    # print(train_miss)

    # val
    # 000273.png 000297.png 000415.png 000468.png 000613.png 000642.png 000746.png 000862.png 000889.png 000917.png 000922.png 000981.png 001050.png 001182.png 001226.png 001260.png 001272.png 001286.png 001389.png 001398.png 001415.png 001589.png 001814.png 001831.png 001844.png 001940.png 002255.png 002365.png 002462.png 002712.png 002726.png 002730.png 002820.png 002827.png 002875.png 002961.png 003134.png 003144.png 003180.png 003225.png 003504.png 003544.png 003620.png 003793.png 004291.png 004807.png 004839.png 004846.png 004958.png 004966.png 005559.png 005572.png 005751.png 005909.png 005994.png 006107.png 006186.png 006560.png 006666.png 006732.png 006850.png 006866.png 006873.png 007027.png 007069.png 007088.png 007204.png 007271.png 007275.png 007290.png 007299.png 007344.png 007413.png
    

    # --> because depth is 11 accum of lidar +- 5 frames --> train miss 90, val miss 73

    
    



