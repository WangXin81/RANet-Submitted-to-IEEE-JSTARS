import os, random, shutil
import glob

TEST_PERCENTAGE = 50
names = []
def  copyFile(image_dir, image_tar_dir_test, image_tar_dir_train,):
    # pathDir = os.listdir(image_dir)
    image_lists = os.listdir(image_dir)
    lengths_lists = len(image_lists)
    test_names = random.sample(image_lists, int(lengths_lists * TEST_PERCENTAGE / 100))
    for name in image_lists:
        na = name[:-4]
        if name in test_names:
            shutil.copy(image_dir + name, image_tar_dir_test + name)
            # shutil.copy(xml_dir + na + '.xml', xml_tar_dir_test + na + '.xml')
        else:
            shutil.copy(image_dir + name, image_tar_dir_train + name)
            # shutil.copy(xml_dir + na + '.xml', xml_tar_dir_train + na + '.xml')

            print("now {} is being proceeding.".format(name))

if __name__ == '__main__':
    dataset_path = '/home/zippor/桌面/dataset/AID/Images/'
    classes = os.listdir(dataset_path)
    for name in classes:
        image_dir = dataset_path + name + '/'
        image_tar_dir_test = '/home/zippor/桌面/dataset/AID/55/val/' + name + '/'
        image_tar_dir_train = '/home/zippor/桌面/dataset/AID/55/train/' + name + '/'
        if not os.path.isdir(image_tar_dir_test):
           os.makedirs(image_tar_dir_test)
        if not os.path.isdir(image_tar_dir_train):
           os.makedirs(image_tar_dir_train)
        copyFile(image_dir, image_tar_dir_test, image_tar_dir_train)
