'''
split labeled images according to 7 labels(7 symptom);2 labels(norman('N') and abnormal('ABN');5 folders(cross-validation)

PS: if the file is .gvf files, use command like 'ffmpeg -i .gvf ./E/%d.png'to extract png frame by frame.
'''
import os
import shutil
imgs_path = '../../dataset/DataCrohnIPI/val/imgs'
csv_file = "../../dataset/DataCrohnIPI/val/CrohnIPI_description.csv"
imgs_split_by_labels_7 = "imgs_split_by_labels_7"
imgs_split_by_labels_2 = "imgs_split_by_labels_2"

imgs_split_by_folds = "imgs_split_by_folds"
with open("../../dataset/DataCrohnIPI/val/CrohnIPI_description.csv") as f:
    for line in f:
        name,label,fold_num=line.strip().split(',')
        if label=='U>10':
            label='U_bigger_than_10'
        if label!='Label':
            img_path = os.path.join(imgs_path,name)
            label_path=os.path.join(imgs_split_by_labels_7, label)

            # if not os.path.exists(label_path):
            #     os.mkdir(label_path)
            # new_imgs_labels_path=os.path.join(label_path,name)
            # fold_path = os.path.join(imgs_split_by_folds,fold_num)
            # if not os.path.exists(fold_path):
            #     os.mkdir(fold_path)
            # new_imgs_folds_path = os.path.join(fold_path,name)
            # shutil.copyfile(img_path, new_imgs_labels_path)
            # shutil.copyfile(img_path, new_imgs_folds_path)
            for n in range(1,6):
                if label=='N':
                    if fold_num==str(n):
                        new_imgs_labels_path2 = os.path.join(imgs_split_by_labels_2,'fold1','N',name)
                    else:
                        new_imgs_labels_path2 = os.path.join(imgs_split_by_labels_2,'notfold1','N',name)
                else:
                    if fold_num==str(n):
                        new_imgs_labels_path2 = os.path.join(imgs_split_by_labels_2,'fold1','ABN',name)
                    else:
                        new_imgs_labels_path2 = os.path.join(imgs_split_by_labels_2,'notfold1','ABN',name)
                shutil.copyfile(img_path, new_imgs_labels_path2)




