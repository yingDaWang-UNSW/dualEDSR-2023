import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sys import stdout

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img,kernel_size, sigma, n_channel):
    blur = _gaussian_kernel(kernel_size, sigma, n_channel, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
    return img
    
    
def random_croptf2(image, height, width):
    cropped_image = tf.image.random_crop(image, size=[image.shape[0], np.min((height,image.shape[1])), np.min((width,image.shape[2])), image.shape[3]])
    return cropped_image
    
def random_croptf23D(image, height, width, depth):
    x=int(tf.floor(np.random.rand()*(image.shape[0]-height)))
    y=int(tf.floor(np.random.rand()*(image.shape[1]-width)))
    z=int(tf.floor(np.random.rand()*(image.shape[2]-depth)))
    cropped_image=tf.expand_dims(image[x:x+height,y:y+width,z:z+depth,:],0)
    #cropped_image = tf.image.random_crop(image, size=[image.shape[0], np.min((height,image.shape[1])), np.min((width,image.shape[2])), image.shape[3]])
    return cropped_image
    
def random_3axis(Cxyz, args):
    x=int(tf.floor(np.random.rand()*(Cxyz.shape[0])))
    y=int(tf.floor(np.random.rand()*(Cxyz.shape[1])))
    z=int(tf.floor(np.random.rand()*(Cxyz.shape[2])))
    
    stackedImage=tf.concat([tf.expand_dims(tf.squeeze(Cxyz[x,:,:]),2),tf.expand_dims(tf.squeeze(Cxyz[:,y,:]),2),tf.expand_dims(tf.squeeze(Cxyz[:,:,z]),2)],2)
    stackedImage=tf.transpose(stackedImage,[2,1,0])
    stackedImage=tf.expand_dims(stackedImage,3)
    return stackedImage
    
def all_3axis(Cxyz, args):        

    stackedImage=tf.concat([Cxyz,tf.transpose(Cxyz,[1,0,2,3]),tf.transpose(Cxyz,[2,1,0,3])],0)

    return stackedImage
    
def createTrainingCubes2(args,HR,LRxy,batchsize,cropsize,scale):
    # read an HR block and extract the LRxy,LRyz, and LRxz blocks of size itersperepoch*batch,x,y,1
    # permute the block so the lrbc dimension is in the batch dimension
    batchLR = np.zeros([batchsize*args.itersPerEpoch,cropsize,cropsize,1],'float32')
    batchHR = np.zeros([batchsize*args.itersPerEpoch*scale,cropsize*scale,cropsize*scale,1],'float32')
    n=0
    n2=0
    for i in range(args.itersPerEpoch):
        # cycle between xy,yz, and xz for extra data - first version was fucked because batch is explicitly the bc dim but it wasnt in this implementation 
        if np.mod(i,3)==0:
            x=int(np.floor(np.random.rand()*(LRxy.shape[0]-batchsize)))
            y=int(np.floor(np.random.rand()*(LRxy.shape[1]-cropsize)))
            z=int(np.floor(np.random.rand()*(LRxy.shape[2]-cropsize)))
            
            block=np.expand_dims(LRxy[x:x+batchsize,y:y+cropsize,z:z+cropsize],3)
            blockHR=np.expand_dims(HR[x*scale:x*scale+batchsize*scale,y*scale:y*scale+cropsize*scale,z*scale:z*scale+cropsize*scale],3)

        elif np.mod(i,3)==1:
            x=int(np.floor(np.random.rand()*(LRxy.shape[0]-cropsize)))
            y=int(np.floor(np.random.rand()*(LRxy.shape[1]-cropsize)))
            z=int(np.floor(np.random.rand()*(LRxy.shape[2]-batchsize)))
            
            block=np.expand_dims(LRxy[x:x+cropsize,y:y+cropsize,z:z+batchsize],3)
            blockHR=np.expand_dims(HR[x*scale:x*scale+cropsize*scale,y*scale:y*scale+cropsize*scale,z*scale:z*scale+batchsize*scale],3)
            block=np.transpose(block,[2,0,1,3])
            blockHR=np.transpose(blockHR,[2,0,1,3])
            

        elif np.mod(i,3)==2:
            x=int(np.floor(np.random.rand()*(LRxy.shape[0]-cropsize)))
            y=int(np.floor(np.random.rand()*(LRxy.shape[1]-batchsize)))
            z=int(np.floor(np.random.rand()*(LRxy.shape[2]-cropsize)))
            
            block=np.expand_dims(LRxy[x:x+cropsize,y:y+batchsize,z:z+cropsize],3)
            blockHR=np.expand_dims(HR[x*scale:x*scale+cropsize*scale,y*scale:y*scale+batchsize*scale,z*scale:z*scale+cropsize*scale],3)
            
            block=np.transpose(block,[1,0,2,3])
            blockHR=np.transpose(blockHR,[1,0,2,3])
            
        batchLR[n:n+batchsize]=block/127.5-1
        batchHR[n2:n2+batchsize*scale]=blockHR/127.5-1
        #batchLR[n:n+batchsize]=block*2-1
        #batchHR[n:n+batchsize]=blockHR*2-1
        n=n+batchsize
        n2=n2+batchsize*scale
        
        stdout.write("\rHR Cube: %d of %d" % (i+1, args.itersPerEpoch))
        stdout.flush()
    stdout.write("\n")
    return batchHR,batchLR

def createTrainingCubesUnpaired(args,LR,BC):
    # read an HR block and extract the LRxy,LRyz, and LRxz blocks of size itersperepoch*batch,x,y,1
    # permute the block so the lrbc dimension is in the batch dimension
    batchLR = np.zeros([args.batch_size*args.itersPerEpoch,args.fine_size,args.fine_size,1],'float32')
    batchBC = np.zeros([args.batch_size*args.itersPerEpoch,args.fine_size,args.fine_size,1],'float32')
    n=0
    #n2=0
    for i in range(args.itersPerEpoch):
        # cycle between xy,yz, and xz for extra data - first version was fucked because batch is explicitly the bc dim but it wasnt in this implementation 
        if np.mod(i,3)==0:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.batch_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.fine_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.batch_size,y:y+args.fine_size,z:z+args.fine_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.batch_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.fine_size)))
            
            blockBC=np.expand_dims(BC[x:x+args.batch_size,y:y+args.fine_size,z:z+args.fine_size],3)

        elif np.mod(i,3)==1:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.batch_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.fine_size,y:y+args.fine_size,z:z+args.batch_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.batch_size)))
            blockBC=np.expand_dims(BC[x:x+args.fine_size,y:y+args.fine_size,z:z+args.batch_size],3)
            
            blockLR=np.transpose(blockLR,[2,0,1,3])
            blockBC=np.transpose(blockBC,[2,0,1,3])
            

        elif np.mod(i,3)==2:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.batch_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.fine_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.fine_size,y:y+args.batch_size,z:z+args.fine_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.batch_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.fine_size)))
            blockBC=np.expand_dims(BC[x:x+args.fine_size,y:y+args.batch_size,z:z+args.fine_size],3)
            
            blockLR=np.transpose(blockLR,[1,0,2,3])
            blockBC=np.transpose(blockBC,[1,0,2,3])
            
        batchLR[n:n+args.batch_size]=blockLR/127.5-1
        batchBC[n:n+args.batch_size]=blockBC/127.5-1
#        batchHR[n2:n2+args.batch_size*scale]=blockHR/127.5-1
        #batchLR[n:n+args.batch_size]=block*2-1
        #batchHR[n:n+args.batch_size]=blockHR*2-1
        n=n+args.batch_size
        #n2=n2+args.batch_size*scale
        
        stdout.write("\rCube: %d of %d" % (i+1, args.itersPerEpoch))
        stdout.flush()
    stdout.write("\n")
    return batchLR,batchBC

def createTrainingCubesUnpaired3D(args,LR,BC):
    # read an HR block and extract the LRxy,LRyz, and LRxz blocks of size itersperepoch*batch,x,y,1
    # permute the block so the lrbc dimension is in the batch dimension
    batchLR = np.zeros([args.batch_size*args.itersPerEpoch,args.fine_size,args.fine_size,args.fine_size,1],'float32')
    batchBC = np.zeros([args.batch_size*args.itersPerEpoch,args.fine_size,args.fine_size,args.fine_size,1],'float32')
    n=0
    #n2=0
    for i in range(args.batch_size*args.itersPerEpoch):
        # cycle between xy,yz, and xz for extra data - first version was fucked because batch is explicitly the bc dim but it wasnt in this implementation 
        if np.mod(i,3)==0:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.batch_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.fine_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.batch_size,y:y+args.fine_size,z:z+args.fine_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.batch_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.fine_size)))
            
            blockBC=np.expand_dims(BC[x:x+args.batch_size,y:y+args.fine_size,z:z+args.fine_size],3)

        elif np.mod(i,3)==1:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.batch_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.fine_size,y:y+args.fine_size,z:z+args.batch_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.batch_size)))
            blockBC=np.expand_dims(BC[x:x+args.fine_size,y:y+args.fine_size,z:z+args.batch_size],3)
            
            blockLR=np.transpose(blockLR,[2,0,1,3])
            blockBC=np.transpose(blockBC,[2,0,1,3])
            

        elif np.mod(i,3)==2:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.batch_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.fine_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.fine_size,y:y+args.batch_size,z:z+args.fine_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.batch_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.fine_size)))
            blockBC=np.expand_dims(BC[x:x+args.fine_size,y:y+args.batch_size,z:z+args.fine_size],3)
            
            blockLR=np.transpose(blockLR,[1,0,2,3])
            blockBC=np.transpose(blockBC,[1,0,2,3])
            
        batchLR[n]=blockLR/127.5-1
        batchBC[n]=blockBC/127.5-1
#        batchHR[n2:n2+args.batch_size*scale]=blockHR/127.5-1
        #batchLR[n:n+args.batch_size]=block*2-1
        #batchHR[n:n+args.batch_size]=blockHR*2-1
        n=n+1
        #n2=n2+args.batch_size*scale
        
        stdout.write("\rCube: %d of %d" % (i+1, args.itersPerEpoch))
        stdout.flush()
    stdout.write("\n")
    return batchLR,batchBC



def createCycleSRTrainingCubesUnpaired3D(args,LR,BC,HR):
    # read an HR block and extract the LRxy,LRyz, and LRxz blocks of size itersperepoch*batch,x,y,1
    # permute the block so the lrbc dimension is in the batch dimension
    batchLR = np.zeros([args.batch_size*args.itersPerEpoch,args.fine_size,args.fine_size,1],'float32')
    batchBC = np.zeros([args.batch_size*args.itersPerEpoch,args.fine_size,args.fine_size,1],'float32')
    batchHR = np.zeros([args.batch_size*args.itersPerEpoch*args.scale,args.fine_size*args.scale,args.fine_size*args.scale,1],'float32')
    n=0
    n2=0
    for i in range(args.itersPerEpoch):
        # cycle between xy,yz, and xz for extra data - first version was fucked because batch is explicitly the bc dim but it wasnt in this implementation 
        if np.mod(i,3)==0:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.batch_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.fine_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.batch_size,y:y+args.fine_size,z:z+args.fine_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.batch_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.fine_size)))
            
            blockBC=np.expand_dims(BC[x:x+args.batch_size,y:y+args.fine_size,z:z+args.fine_size],3)
            blockHR=np.expand_dims(HR[x*args.scale:x*args.scale+args.batch_size*args.scale,y*args.scale:y*args.scale+args.fine_size*args.scale,z*args.scale:z*args.scale+args.fine_size*args.scale],3)
        elif np.mod(i,3)==1:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.batch_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.fine_size,y:y+args.fine_size,z:z+args.batch_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.fine_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.batch_size)))
            blockBC=np.expand_dims(BC[x:x+args.fine_size,y:y+args.fine_size,z:z+args.batch_size],3)
            blockHR=np.expand_dims(HR[x*args.scale:x*args.scale+args.fine_size*args.scale,y*args.scale:y*args.scale+args.fine_size*args.scale,z*args.scale:z*args.scale+args.batch_size*args.scale],3)
            
            blockLR=np.transpose(blockLR,[2,0,1,3])
            blockBC=np.transpose(blockBC,[2,0,1,3])
            blockHR=np.transpose(blockHR,[2,0,1,3])

        elif np.mod(i,3)==2:
            x=int(np.floor(np.random.rand()*(LR.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(LR.shape[1]-args.batch_size)))
            z=int(np.floor(np.random.rand()*(LR.shape[2]-args.fine_size)))
            
            blockLR=np.expand_dims(LR[x:x+args.fine_size,y:y+args.batch_size,z:z+args.fine_size],3)
            
            x=int(np.floor(np.random.rand()*(BC.shape[0]-args.fine_size)))
            y=int(np.floor(np.random.rand()*(BC.shape[1]-args.batch_size)))
            z=int(np.floor(np.random.rand()*(BC.shape[2]-args.fine_size)))
            blockBC=np.expand_dims(BC[x:x+args.fine_size,y:y+args.batch_size,z:z+args.fine_size],3)
            blockHR=np.expand_dims(HR[x*args.scale:x*args.scale+args.fine_size*args.scale,y*args.scale:y*args.scale+args.batch_size*args.scale,z*args.scale:z*args.scale+args.fine_size*args.scale],3)
            blockLR=np.transpose(blockLR,[1,0,2,3])
            blockBC=np.transpose(blockBC,[1,0,2,3])
            blockHR=np.transpose(blockHR,[1,0,2,3])
            
        batchLR[n:n+args.batch_size]=blockLR/127.5-1
        batchBC[n:n+args.batch_size]=blockBC/127.5-1
        batchHR[n2:n2+args.batch_size*args.scale]=blockHR/127.5-1
#        batchHR[n2:n2+args.batch_size*scale]=blockHR/127.5-1
        #batchLR[n:n+args.batch_size]=block*2-1
        #batchHR[n:n+args.batch_size]=blockHR*2-1
        n=n+args.batch_size
        n2=n2+args.batch_size*args.scale
        
        stdout.write("\rCube: %d of %d" % (i+1, args.itersPerEpoch))
        stdout.flush()
    stdout.write("\n")
    return batchLR,batchBC,batchHR



def augmentData(image):
    # inject random contrast and brightness adjustments
    contFactor = (np.random.rand()*2-1)*0.2+1
    brightFactor = (np.random.rand()*2-1)*0.2+1

    image = image*brightFactor

    image = (image-tf.math.reduce_mean(image))*contFactor + tf.math.reduce_mean(image)
    image = tf.clip_by_value(image,-1,1)
    return image
