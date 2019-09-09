from .libraries import *
from .tfrecords import *


def normalize_image(x,y):
  return (x['image']/255, y)


@tf.function
def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3,h=-1,w=-1):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    #HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channel = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)
    if h==-1:
        erase_area_low_bound = tf.cast(tf.round(tf.sqrt(sl * area * r1)), tf.int32)
        erase_area_up_bound = tf.cast(tf.round(tf.sqrt((sh * area) / r1)), tf.int32)
        h_upper_bound = tf.minimum(erase_area_up_bound, height)
        w_upper_bound = tf.minimum(erase_area_up_bound, width)
        h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
        w = tf.random.uniform([], erase_area_low_bound, w_upper_bound, tf.int32)
    else:
        h =  arg = tf.convert_to_tensor(h, dtype=tf.int32)
        w =  arg = tf.convert_to_tensor(w, dtype=tf.int32)
    x1 = tf.random.uniform([], 0, height+1 - h, tf.int32)
    y1 = tf.random.uniform([], 0, width+1 - w, tf.int32)
    erase_area = tf.cast(tf.random.uniform([h, w, channel], 0, 255, tf.int32), tf.uint8)
    erasing_img = img[x1:x1+h, y1:y1+w, :].assign(erase_area)
    return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: img, lambda: erasing_img)



def data_aug(ds,class_names,class_weight,augment_weight,out_file):
  def random_crop(x,y,params):
    return (tf.image.random_crop(x, params['output_dim']),y)
  def flip_LR(x,y,params):
    return (tf.image.flip_left_right(x),y)


  fn_map={'flip_LR':flip_LR,'random_crop':random_crop}
  First=True
  ctr=0
  for i in ds:
    aug_choice=(np.random.randint(1,101,len(class_weight)*len(augment_weight))/100).reshape(len(class_weight),len(augment_weight))
    class_choice=(np.random.randint(1,101,len(class_weight))/100)
    ctr=ctr+1
    x=i[0].numpy()
    y=i[1].numpy()

    a_ch=list(aug_choice[y])
    cw=class_weight[class_names[y]]
    aw=[i['weight'] for i in augment_weight.values()]

    f_list=[list(augment_weight.keys())[i] for i in range(len(a_ch)) if class_choice[i]<=cw and a_ch[i]<=aw[i]]
    # if f_list!=[]:
    #   print('\n')
    for f in f_list:
      f1=fn_map[f]
      k=f1(x,y,augment_weight[f]['params'])
      # print(class_choice[i],a_ch,ctr,f.__name__)
      print('Augmenting Image ',ctr,' with ',f)
      x1=(k[0].numpy()).reshape(1,k[0].shape[0],k[0].shape[1],k[0].shape[2])
      y1=np.array([k[1]])
      if First:
        x_augmented=x1
        y_augmented=y1
        First=False
      else:
        x_augmented=np.vstack((x_augmented,x1))
        y_augmented=np.vstack((y_augmented,y1))
      # print(x_augmented.shape)
      # print(y_augmented.shape)
    # print("***************************************************************************************************")
  create_tfrecords(out_file,(x_augmented, y_augmented))

@timer
# @checkTFversion
def augment_images(path,class_weight,augment_weight,parallelize):
  class_names = [line.rstrip('\n') for line in open(os.path.join(path, "classes.txt"))]
  in_file=os.path.join(path, "train.tfrecords")
  ds=tf.data.TFRecordDataset(tf.data.Dataset.list_files(in_file))
  ds=ds.map(lambda record: parser(record), num_parallel_calls=parallelize)
  ds=ds.map(normalize_image, num_parallel_calls=parallelize)
  out_file=os.path.join(path, "train_aug.tfrecords")
  data_aug(ds,class_names,class_weight,augment_weight,out_file)
