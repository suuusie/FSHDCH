import scipy.io as sio
import h5py
# import tables

def loading_data(path):

    # FashionVC
    image_path = path + "image.mat"
    label_path = path + "label.mat"
    tag_path = path + "tag.mat"

    # images = h5py.File(image_path)['Image'][:]   # 18694,224,224,3
    # images = images.transpose(3,2,1,0)
    images = sio.loadmat(image_path)['Image']   # 19862,224,224,3
    tags = sio.loadmat(tag_path)['Tag']     # 19862,2685
    labels = sio.loadmat(label_path)['Label']    # 19862,35

    return images, tags, labels


if __name__ == '__main__':
    path = '/data/from145/Hash/hierarchical hashing/SHDCH/DataSet/FashionVC/'
