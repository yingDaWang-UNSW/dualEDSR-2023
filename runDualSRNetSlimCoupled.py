'''
Double 2D super resolution method
'''
#run runDualSRNetSlimCoupled.py --dataset_dir '/media/user/HDD5/Dual-EDSR-main/dataset for training/' --gpuIDs 1 --modelName 'dualEDSR-2023-imperialCarbonate' --continue_train False --continueEpoch 24
#   ...: 0  --batch_size 32 --fine_size 32 --augFlag False --itersPerEpoch 333 --iterCyclesPerEpoch 3 --scale 4 --save_freq 1 --srganFlag False --phase 'test' --test_dir '/media/user/HDD5/Dual-EDSR-main/' --t
#   ...: est_temp_save_dir '/media/user/SSD2/allP/temp/' --test_save_dir '/media/user/HDD5/Dual-EDSR-main/testSR' --valNum 0 --print_freq 10 --ngsrf 64 --srAdv_lambda 1e-3


#TODO: write up testing section if train if test. enable substacking!
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
import netsLayersLosses
import IOAugmentations
import tifffile
# Helper libraries
from sys import stdout
import numpy as np
import os
from glob import glob
import time
import datetime
import pdb
import imageio
from matplotlib import pyplot as plt
import h5py

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

import argparse

def args():
    # TODO modularise this into concentricGAN noise->clean->SR->seg->vels
    parser = argparse.ArgumentParser(description='')
    #training arguments
    parser.add_argument('--mixedPrecision', dest='mixedPrecision', type=str2bool, default=False, help='16bit computes')
    parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='2', help='IDs for the GPUs. Empty for CPU. Nospaces')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/media/user/SSD2/fuelCellDataset2Dxy/', help='dataset path - include last slash')
    parser.add_argument('--augFlag', dest='augFlag', type=str2bool, default=False, help='augmentation')

    parser.add_argument('--scale', dest='scale', type=str2int, default=4, help='sr scale factor')
    parser.add_argument('--batch_size', dest='batch_size', type=str2int, default=64, help='# 2D images in subbatch')
    parser.add_argument('--subBlocks', dest='subBlocks', type=str2int, default=4, help='# 3D images in batch')
    parser.add_argument('--fine_size', dest='fine_size', type=str2int, default=64, help='then crop LR to this size')

    
    parser.add_argument('--epoch', dest='epoch', type=str2int, default=500, help='# of epoch')        
    parser.add_argument('--itersPerEpoch', dest='itersPerEpoch', type=str2int, default=300, help='# iterations per epoch') 
    parser.add_argument('--iterCyclesPerEpoch', dest='iterCyclesPerEpoch', type=str2int, default=3, help='# iteration cycles per epoch') 

    parser.add_argument('--valNum', dest='valNum', type=str2int, default=10, help='# max val images') 

    # base model uses dualEDSR
    parser.add_argument('--ngsrf', dest='ngsrf', type=str2int, default=32, help='# of gen SR filters in first conv layer')
    parser.add_argument('--numResBlocks', dest='numResBlocks', type=str2int, default=16, help='# of resBlocks in SR')
    parser.add_argument('--segFlag', dest='segFlag', type=str2bool, default=False, help='segFlag') 
    parser.add_argument('--numChannels', dest='numChannels', type=str2int, default=1, help='numChannels')
    # extra loss functions

    # base model uses SCGAN
    parser.add_argument('--srganFlag', dest='srganFlag', type=str2bool, default=False, help='if gan is active') 
    parser.add_argument('--ndsrf', dest='ndsrf', type=str2int, default=32, help='# of disc SR filters in first conv layer')
    parser.add_argument('--srAdv_lambda', dest='srAdv_lambda', type=str2float, default=1e-2, help='weight on Adv term for normal sr')
    
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--epoch_step', dest='epoch_step', type=str2int, default=50, help='# of epoch to decay lr')

    parser.add_argument('--phase', dest='phase', type=str, default='train', help='train, test')
    
    # Model IO
    parser.add_argument('--save_freq', dest='save_freq', type=str2int, default=10, help='save a model every save_freq epochs')
    parser.add_argument('--print_freq', dest='print_freq', type=str2int, default=10, help='print the validation images every X epochs')
    parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--continueEpoch', dest='continueEpoch', type=str2int, default=0, help='')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints', help='models are saved here')
    parser.add_argument('--modelName', dest='modelName', default='dual2DSRTest', help='models are loaded here')
    
    # testing arguments
    parser.add_argument('--test_dir', dest='test_dir', default='/media/user/SSD2/testLR/', help='test sample slices are saved here as png slices')
    parser.add_argument('--test_temp_save_dir', dest='test_temp_save_dir', default='/media/user/SSD2/', help='test sample are saved here')
    parser.add_argument('--test_save_dir', dest='test_save_dir', default='/media/user/SSD2/', help='test sample are saved here')
    args = parser.parse_args()

    return args
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v

args=args() # args is global

gpuList=args.gpuIDs
args.numGPUs = len(gpuList.split(','))
if args.numGPUs<=4:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuList

if args.mixedPrecision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

else:
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# detect hardware
if len(args.gpuIDs.split(','))<=1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# define the network
with strategy.scope():
    # define functions used

    def createSRGenerator(args):

        generator = netsLayersLosses.edsr(scale=args.scale, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks, ndims=2)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator)
        return generator, optimizerGenerator            
        
    def createSRCGenerator(args):

        generator = netsLayersLosses.edsr1D(scale=args.scale, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks, ndims=2)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator)
        return generator, optimizerGenerator            

    def createSRDiscriminator(args):    
        if args.srganFlag:
#            discriminator = netsLayersLosses.DiscriminatorSRGAN3Axis(args)
            discriminator = netsLayersLosses.DiscriminatorSRGAN3D(args)
            discriminator.summary(200)
        else:
            a = tf.keras.layers.Input(shape=(1,))
            b = a
            discriminator = tf.keras.models.Model(inputs=a, outputs=b)
        optimizerDiscriminator = tf.keras.optimizers.Adam(learning_rate=args.lr)           
        optimizerDiscriminator = mixed_precision.LossScaleOptimizer(optimizerDiscriminator)     
        return discriminator, optimizerDiscriminator
        

    # define the actions taken per iteration (calc grads and make an optim step)
    def train_step(HRBatch,BCBatch):
        Cxyz, Bxy =  HRBatch, BCBatch # make sure the dims are correct
        if args.augFlag:
            Bxy = IOAugmentations.augmentData(Bxy)
        # train
        with tf.GradientTape(persistent=True) as tape:
            # run a cycle on the cycleGAN
            totalGsrXYLoss = 0
            totalGsrYZLoss = 0
            
            
            advsrLoss = 0
            totalDsrLoss = 0

            
            Cxyd=tf.image.resize(tf.squeeze(Cxyz),[Cxyz.shape[0]//args.scale,Cxyz.shape[2]],method='bicubic')
            Cxyd=tf.expand_dims(Cxyd,3)
            SRxy = generatorSR(Bxy, training=True)
            totalGsrXYLoss = netsLayersLosses.meanAbsoluteError(Cxyd, SRxy)
                
            # set bit depth to 8 for SRxy
            SRxy=(SRxy+1)*127.5
            SRxy=tf.math.round(SRxy)
            SRxy=SRxy/127.5 - 1
            # transpose the volume
            SRxy = tf.transpose(SRxy,perm=[1,0,2,3])
            Cxyz = tf.transpose(Cxyz,perm=[1,0,2,3])
            
            # resize the slices
            # SRxyd=tf.image.resize(SRxy,[SRxy.shape[1],SRxy.shape[2]//args.scale],method='bicubic')
            
            SRxyz = generatorSRC(SRxy, training=True)
            totalGsrYZLoss = netsLayersLosses.meanAbsoluteError(Cxyz, SRxyz)
         
            if args.srganFlag:
#                disc_C = discriminatorSR(IOAugmentations.all_3axis(Cxyz, args), training=True)
#                disc_BASR = discriminatorSR(IOAugmentations.all_3axis(SRxyz, args), training=True)
                
                disc_C = discriminatorSR(tf.expand_dims(Cxyz,0), training=True)
                disc_BASR = discriminatorSR(tf.expand_dims(SRxyz, 0), training=True)
                # use relgan here
#                advsrLoss = advsrLoss + netsLayersLosses.rel_advScganLoss(disc_C, disc_BASR)
#                totalDsrLoss = totalDsrLoss + netsLayersLosses.rel_scganLoss(disc_C, disc_BASR)
                
                advsrLoss = advsrLoss + netsLayersLosses.advScganLoss(disc_BASR)
                totalDsrLoss = totalDsrLoss + netsLayersLosses.scganLoss(disc_C, disc_BASR)
                
            totalGsrYZLoss = totalGsrYZLoss + totalGsrXYLoss + 0.5*args.srAdv_lambda*advsrLoss
            
            totalGsrXYZLoss = totalGsrYZLoss + 0.5*args.srAdv_lambda*advsrLoss
                        
                        
            if args.srganFlag:
                totalDsrLossScal = optimizerDiscriminatorSR.get_scaled_loss(totalDsrLoss)
                
            totalGsrLossScal = optimizerGeneratorSR.get_scaled_loss(totalGsrYZLoss)
            totalGsrcLossScal = optimizerGeneratorSRC.get_scaled_loss(totalGsrXYZLoss)
                
        # calculate gradients
        gradGsr = tape.gradient(totalGsrLossScal, generatorSR.trainable_variables)
        gradGsrc = tape.gradient(totalGsrcLossScal, generatorSRC.trainable_variables)

        # unscale gradients
        gradGsr = optimizerGeneratorSR.get_unscaled_gradients(gradGsr)
        gradGsrc = optimizerGeneratorSRC.get_unscaled_gradients(gradGsrc)
            
        # apply gradients
        optimizerGeneratorSR.apply_gradients(zip(gradGsr,generatorSR.trainable_variables))
        optimizerGeneratorSRC.apply_gradients(zip(gradGsrc,generatorSRC.trainable_variables))
        
        if args.srganFlag:
            gradDsr = tape.gradient(totalDsrLossScal, discriminatorSR.trainable_variables)
            gradDsr = optimizerDiscriminatorSR.get_unscaled_gradients(gradDsr)
            optimizerDiscriminatorSR.apply_gradients(zip(gradDsr,discriminatorSR.trainable_variables))

            
        return totalGsrXYLoss, totalGsrYZLoss, advsrLoss, totalDsrLoss

   
    @tf.function
    def distributed_train_step(HRBatch,BCBatch):
        PRGABL, PRGBAL, PRADVSRL, PRDSRL = strategy.run(train_step, args=(HRBatch,BCBatch))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, PRGABL, axis=None),  strategy.reduce(tf.distribute.ReduceOp.SUM, PRGBAL, axis=None),  strategy.reduce(tf.distribute.ReduceOp.SUM, PRADVSRL, axis=None),  strategy.reduce(tf.distribute.ReduceOp.SUM, PRDSRL, axis=None)
        
    # begin actual script here
    generatorSR, optimizerGeneratorSR = createSRGenerator(args)
    generatorSRC, optimizerGeneratorSRC = createSRCGenerator(args)
    discriminatorSR, optimizerDiscriminatorSR = createSRDiscriminator(args)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(f'{args.checkpoint_dir}/training_outputs'):
        os.mkdir(f'{args.checkpoint_dir}/training_outputs')
    trainingDir=f"{args.checkpoint_dir}/{args.modelName}/"
    initEpoch=0
    if args.continue_train or args.phase == 'test': # restore the weights if requested, or if testing
        print(f'Loading checkpoints from {trainingDir} for epoch {args.continueEpoch}')
        try:
            generatorSR.load_weights(f'{trainingDir}/GSR-{args.continueEpoch}/GSR')
            generatorSRC.load_weights(f'{trainingDir}/GSRC-{args.continueEpoch}/GSRC')
            initEpoch=args.continueEpoch
        except:
            print('Could not load SR related weights')
            
        if args.srganFlag:
            try:
                discriminatorSR.load_weights(f'{trainingDir}/DSR-{args.continueEpoch}/DSR')
            except:
                print('Could not load SRGAN related weights')
    # run
    if args.phase == 'train':
        EPOCHS = args.epoch
        valoutDir = args.dataset_dir.split('/')[-2]
        # Create a checkpoint directory to store the checkpoints.
        rightNow=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trainOutputDir=f'{args.checkpoint_dir}/training_outputs/{rightNow}-distNN-{valoutDir}-{args.modelName}/'
        if not os.path.exists(trainingDir):
            os.mkdir(trainingDir)
        os.mkdir(trainOutputDir)

        print('2D/3D training specified, datasets will be randomly mini-batched per epoch')
        print('2D/3D dataset and training -> data will be fully preloaded into RAM')
        
        BCLoc=glob(args.dataset_dir+'LR/LR.npy')
        LRxy=np.load(BCLoc[0], mmap_mode='r')
        #LRxy=np.transpose(LRxy,[2,1,0])

        
        HRLoc=glob(args.dataset_dir+'HR/HR.npy')
        HR=np.load(HRLoc[0], mmap_mode='r')
        #HR=np.transpose(HR,[2,1,0])

        start_time = time.time()
        epoch=initEpoch
        while epoch <EPOCHS:
            if args.srganFlag:
                batchSizeThisEpoch = args.batch_size
                fineSizeThisEpoch = args.fine_size
            else:
                totalPerBatchVoxels=args.fine_size*args.fine_size*args.batch_size
                minPerDimSize=args.scale*2
                maxPerDimSize=args.fine_size
                batchSizeThisEpoch =int(np.floor(np.random.rand()*(maxPerDimSize-minPerDimSize))+minPerDimSize)
                fineSizeThisEpoch = int(np.floor(np.sqrt(totalPerBatchVoxels/batchSizeThisEpoch)))
            print(f'Reading and Distributing Dataset into GPUs, block size this epoch: {batchSizeThisEpoch} x {fineSizeThisEpoch} x {fineSizeThisEpoch} -> {args.scale}x')
            realHRBatches, realBCBatches = IOAugmentations.createTrainingCubes2(args,HR,LRxy,batchSizeThisEpoch,fineSizeThisEpoch, args.scale)           
           
            HR_dataset = tf.data.Dataset.from_tensor_slices((realHRBatches)).batch(batchSizeThisEpoch*args.scale) 
            HR_dataset_dist = strategy.experimental_distribute_dataset(HR_dataset)
            
            HR_dataset_test=tf.data.Dataset.from_tensor_slices((realHRBatches[0:args.valNum*batchSizeThisEpoch*args.scale])).batch(batchSizeThisEpoch*args.scale) 
            
            
            LR_dataset = tf.data.Dataset.from_tensor_slices((realBCBatches)).batch(batchSizeThisEpoch) 
            LR_dataset_dist = strategy.experimental_distribute_dataset(LR_dataset)
            
            LR_dataset_test=tf.data.Dataset.from_tensor_slices((realBCBatches[0:args.valNum*batchSizeThisEpoch])).batch(batchSizeThisEpoch) 
            # TRAIN LOOP
            lastTime=time.time()

            lr=args.lr * 0.5**(epoch/args.epoch_step) # add cosine annealing later

            optimizerGeneratorSR.learning_rate = lr
            optimizerGeneratorSRC.learning_rate = lr
            totGABL = 0
            totGBAL = 0
            totADVSRL = 0
            totDSRL = 0
            num_batches = 0
            numSkips=0;
            print(f'Learning Rate: {lr:.4e}')
            while num_batches < args.itersPerEpoch*args.iterCyclesPerEpoch:
                for x, y in zip(HR_dataset, LR_dataset):
                    num_batches += 1
                    GABL, GBAL, ADVSRL, DSRL = distributed_train_step(x, y)
                    totGABL += GABL
                    totGBAL += GBAL
                    totADVSRL += ADVSRL
                    totDSRL += DSRL
                    currentTime=time.time()
                    
                    
                    stdout.write("\rEpoch: %4d, Iter: %4d, Time: %4.4f, Speed: %4.4f its/s, GSRxyL: %4.4f, GSRyzL: %4.4f, advSRL: %4.4f, DSRL: %4.4f" % (epoch+1, num_batches, currentTime-start_time, 1/(currentTime-lastTime), GABL, GBAL, ADVSRL, DSRL))
                    stdout.flush()
                    lastTime=currentTime

            stdout.write("\n")
            num_batches=num_batches-numSkips
            totGABL /= num_batches
            totGBAL /= num_batches
            totADVSRL /= num_batches
            totDSRL /= num_batches
            print('Mean Epoch Performance: GSRxyL: %4.4f, GSRyzL: %4.4f, advSRL: %4.4f, DSRL: %4.4f' % (totGABL, totGBAL, totADVSRL, totDSRL))
            
            if np.mod(epoch+1, args.print_freq) == 0 or epoch == 0:
                # validation LOOP
                # validation LOOP
                # run a real validation on the LR block
                testFiles = sorted(glob(args.test_dir+'/*.npy'))          
                for testFile in testFiles:
                    #testFile=testFiles[0]
                    print(f'Super Resolving {testFile}')
                    testFileName=testFile.split('/')[-1]
                    domain=np.load(testFile, mmap_mode='r')
                    domain=domain[0:200,0:200,0:200]
                    domain=np.transpose(domain,[2,0,1])
                    domain=domain[0:domain.shape[0]//8*8,0:domain.shape[1]//8*8,0:domain.shape[2]//8*8] 
                    domainSRxy=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]],'uint8')
                    domainSRxyz=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]*args.scale],'uint8')
                    for i in range(domain.shape[2]):
                        slicez=domain[:,:,i]
                        slicez = (slicez/127.5) - 1 # block will auto cast to float, thanks python
                        slicez=tf.cast(slicez, tf.float32)
                        slicez=tf.expand_dims(slicez,2)
                        slicez=tf.expand_dims(slicez,0)
                        stdout.write("\rXY Pass: Super Resolving slice %d" % (i))
                        stdout.flush()

                        ABsr=generatorSR(slicez)
                        ABsr=np.asarray(ABsr)
                        ABsr=np.squeeze(ABsr)
                        ABsr=(ABsr+1)*127.5
                        ABsr=tf.math.round(ABsr)
                        ABsr=np.asarray(ABsr,'uint8')
                        maxSR=ABsr
                        domainSRxy[:,:,i]=maxSR
                    stdout.write("\n")
                    tifffile.imwrite(f'{args.test_save_dir}/{testFileName}-SRxy-{args.scale}x-{args.modelName}-{epoch}.tif', np.squeeze(domainSRxy))
                    for x in range(domain.shape[0]*args.scale):
                        slicex=domainSRxy[x,:,:]
                        slicex=slicex/127.5 - 1
                        slicex=tf.transpose(slicex,[1,0])
                        slicex=tf.expand_dims(slicex,2)
                        slicex=tf.expand_dims(slicex,0)

                        stdout.write("\rYZ Pass: Super Resolving slice %d" % (x))
                        stdout.flush()
                        ABsr=generatorSRC(slicex)
                        ABsr=np.squeeze(ABsr)
                        ABsr=(ABsr+1)*127.5
                        ABsr=np.asarray(ABsr,'uint8')
                        maxSR=ABsr
                        maxSR=tf.transpose(maxSR,[1,0])
                        domainSRxyz[x,:,:]=np.asarray(maxSR,'uint8')
                    stdout.write("\n")
                    tifffile.imwrite(f'{args.test_save_dir}/{testFileName}-SRxyz-{args.scale}x-{args.modelName}-{epoch}.tif', np.squeeze(domainSRxyz))
                if args.valNum>0:
                    valPSNRC=0.0
                    valPSNRCC=0.0
                    
                    numTestBatches=0
                    os.mkdir(f'./{trainOutputDir}/epoch-{epoch+1}/')

                    for C, B in zip(HR_dataset_test, LR_dataset_test):

                        #B = BC[0][1]
                        #C = BC[0][0]

                        Cd = tf.image.resize(tf.squeeze(C),[C.shape[0]//args.scale,C.shape[2]],method='bicubic')
                        Cd=tf.expand_dims(Cd,3)
                        Co = np.asarray(Cd)
                        fakeC = generatorSR(B, training=False)
                        fakeCo = np.asarray(fakeC)
                        
                        psnrC=tf.image.psnr(fakeC,Cd,2)
                        # set bit depth to 8 for SRxy
                        fakeC=(fakeC+1)*127.5
                        fakeC=tf.math.round(fakeC)
                        fakeC=fakeC/127.5 - 1
                        # transpose and downsample here
                        fakeC = tf.transpose(fakeC,[1,0,2,3])
                        B = tf.transpose(B,[1,0,2,3])
                        C = tf.transpose(C,[1,0,2,3])
                        #fakeC=tf.image.resize(fakeC,[fakeC.shape[1],fakeC.shape[2]//args.scale],method='bicubic')
                        fakeC_clean = generatorSRC(fakeC, training=False)
                        psnrCC=tf.image.psnr(fakeC_clean,C,2)
                        
                        B = np.asarray(B)
                        C = np.asarray(C)
                        fakeC = np.asarray(fakeC)
                        fakeC_clean = np.asarray(fakeC_clean)

                        valPSNRC += np.mean(psnrC)
                        valPSNRCC += np.mean(psnrCC)
                        numTestBatches += 1

                        
                        image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Bxy.tif'
                        B=(B+1)*127.5
                        tifffile.imwrite(image_path, np.array(np.squeeze(B.astype('uint8')), dtype='uint8'))

                        image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Cxyz.tif'
                        Co=(Co+1)*127.5
                        tifffile.imwrite(image_path, np.array(np.squeeze(Co.astype('uint8')), dtype='uint8'))
                        
                        image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Ctxyz.tif'
                        C=(C+1)*127.5
                        tifffile.imwrite(image_path, np.array(np.squeeze(C.astype('uint8')), dtype='uint8'))

                        image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxy.tif'
                        fakeCo=(fakeCo+1)*127.5
                        tifffile.imwrite(image_path, np.array(np.squeeze(fakeCo.astype('uint8')), dtype='uint8'))

                        image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxytd.tif'
                        fakeC=(fakeC+1)*127.5
                        tifffile.imwrite(image_path, np.array(np.squeeze(fakeC.astype('uint8')), dtype='uint8'))
                        
                        image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxyz.tif'
                        fakeC_clean=(fakeC_clean+1)*127.5
                        tifffile.imwrite(image_path, np.array(np.squeeze(fakeC_clean.astype('uint8')), dtype='uint8'))
                            

                        
                        stdout.write("\rIter: %4d, Test: PSNR-SR: %4.4f, PSNR-SRC: %4.4f" %(numTestBatches, np.mean(psnrC), np.mean(psnrCC)))
                        stdout.flush()
                        if numTestBatches == args.valNum:
                            break

                    valPSNRC /= numTestBatches
                    valPSNRCC /= numTestBatches

                    stdout.write("\n")
                    print(f'Mean Validation PSNR-SR: {valPSNRC}, PSNR-SRC: {valPSNRCC}')
                        
            if (epoch) % args.save_freq == 0:
                #checkpoint.save(checkpoint_prefix)
                print('Saving network weights (archive)')
                generatorSR.save_weights(f'{trainingDir}/GSR-{epoch}/GSR')
                generatorSRC.save_weights(f'{trainingDir}/GSRC-{epoch}/GSRC')
                if args.srganFlag:
                    discriminatorSR.save_weights(f'{trainingDir}/DSR-{epoch}/DSR')
                        
                print('Saving network weights (rewritable checkpoint)')
                generatorSR.save_weights(f'{trainingDir}/GSR/GSR')
                generatorSRC.save_weights(f'{trainingDir}/GSRC/GSRC')
                if args.srganFlag:
                    discriminatorSR.save_weights(f'{trainingDir}/DSR/DSR')
                    
                print('Saving model (rewritable checkpoint)')
                generatorSR.save(f'{trainingDir}/GSR-{epoch}.h5')
                generatorSRC.save(f'{trainingDir}/GSRC-{epoch}.h5')
                if args.srganFlag:
                    discriminatorSR.save(f'{trainingDir}/DSR-{epoch}.h5')
            epoch=epoch+1

    elif args.phase == 'testSmall':
        # test within scope?
        # read entire LR block of size x,y,zb and upscale to xs,ys,zs

        testFiles = sorted(glob(args.test_dir+'/*.npy'))          

        for testFile in testFiles:
            #testFile=testFiles[0]
            print(f'XY Pass: Super Resolving {testFile}')
            
            domain=np.load(testFile)
            domain=domain[0:domain.shape[0]//8*8,0:domain.shape[1]//8*8,0:domain.shape[2]//8*8] 
            domainSRxy=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]],'uint8')
            domainSRxyz=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]*args.scale],'uint8')
            
            for i in range(domain.shape[2]):
                slicez=domain[:,:,i]
                slicez = (slicez/127.5) - 1 # block will auto cast to float, thanks python
                slicez=tf.cast(slicez, tf.float32)
                slicez=tf.expand_dims(slicez,2)
                slicez=tf.expand_dims(slicez,0)
                maxNz=550
                dualLength=100
                numParts=slicez.shape[2]//(maxNz-dualLength)+1
                z=0
                zz=0
                maxSR=np.zeros([slicez.shape[1]*args.scale,slicez.shape[2]*args.scale],'uint8')
                if numParts==0:
                    print(f'XY Pass: Super Resolving slice {i}')
                    ABsr=generatorSR(slicez)
                    ABsr=np.asarray(ABsr)
                    ABsr=np.squeeze(ABsr)
                    ABsr=(ABsr+1)*127.5
                    ABsr=tf.math.round(ABsr)
                    ABsr=np.asarray(ABsr,'uint8')
                    maxSR=ABsr
                else:
                    for n in range(numParts):
                        print(f'XY Pass: Super Resolving Slice {i} Subsection {n+1}')
                        tempSlice=slicez[:,:,zz:zz+maxNz]
                        ABsr=generatorSR(tempSlice)
                        ABsr=np.asarray(ABsr)
                        ABsr=np.squeeze(ABsr)
                        ABsr=(ABsr+1)*127.5
                        ABsr=tf.math.round(ABsr)
                        ABsr=np.asarray(ABsr,'uint8')
                        if n==0:
                            maxSR[:,:(z+maxNz-dualLength//2)*args.scale]=ABsr[:,:(maxNz-dualLength//2)*args.scale]
                            z=z+maxNz-dualLength//2
                        elif n==numParts-1:
                            maxSR[:,(z)*args.scale:]=ABsr[:,dualLength//2*args.scale:]
                        else:
                            maxSR[:,(z)*args.scale:(z+maxNz-dualLength)*args.scale]=ABsr[:,dualLength//2*args.scale:(maxNz-dualLength//2)*args.scale]
                            z=z+maxNz-dualLength
                        zz=zz+maxNz-dualLength
                domainSRxy[:,:,i]=maxSR
                if np.mod(i,100)==0:
                    image_path = f'{testFile}-SRxy-{i}-{args.scale}x-{args.modelName}.tif'
                    tifffile.imwrite(image_path, np.squeeze(maxSR))
                
            for x in range(domainSRxy.shape[0]):
                maxNz=550
                dualLength=100
                slicex=domainSRxy[x,:,:]
                slicex=slicex/127.5 - 1
                slicex=tf.transpose(slicex,[1,0])
                slicex=tf.expand_dims(slicex,2)
                slicex=tf.expand_dims(slicex,0)
                
                
                numParts=slicex.shape[1]//(maxNz-dualLength)+1
                z=0
                zz=0
                maxSR=np.zeros([slicex.shape[1]*args.scale,slicex.shape[2]],'uint8')
                if numParts==0:
                    print(f'YZ Pass: Super Resolving slice {x}')

                    ABsr=generatorSRC(slicex)
                    ABsr=np.squeeze(ABsr)
                    ABsr=(ABsr+1)*127.5
                    ABsr=np.asarray(ABsr,'uint8')
                    maxSR=ABsr
                else:
                    for n in range(numParts):
                        print(f'YZ Pass: Super Resolving Slice {x} Subsection {n+1}')
                        tempSlice=slicex[:,zz:zz+maxNz,:]
                        ABsr=generatorSRC(tempSlice)
                        ABsr=np.asarray(ABsr)
                        ABsr=np.squeeze(ABsr)
                        ABsr=(ABsr+1)*127.5
                        ABsr=tf.math.round(ABsr)
                        ABsr=np.asarray(ABsr,'uint8')
                        if n==0:
                            maxSR[:(z+maxNz-dualLength//2)*args.scale,:]=ABsr[:(maxNz-dualLength//2)*args.scale,:]
                            z=z+maxNz-dualLength//2
                        elif n==numParts-1:
                            maxSR[(z)*args.scale:,:]=ABsr[dualLength//2*args.scale:,:]
                        else:
                            maxSR[(z)*args.scale:(z+maxNz-dualLength)*args.scale,:]=ABsr[dualLength//2*args.scale:(maxNz-dualLength//2)*args.scale,:]
                            z=z+maxNz-dualLength
                        zz=zz+maxNz-dualLength
                    
                maxSR=tf.transpose(maxSR,[1,0])
                domainSRxyz[x,:,:]=np.asarray(maxSR,'uint8')
                if np.mod(x,100)==0:
                    image_path = f'{testFile}-SRxyz-{x}-{args.scale}x-{args.modelName}.tif'
                    tifffile.imwrite(image_path, np.squeeze(maxSR))


#            for z in range(domain.shape[2]):
#                print(f'XY Pass: Super Resolving slice {z}')
#                slicez=domain[:,:,z]
#                slicez = (slicez/127.5) - 1 # block will auto cast to float, thanks python
#                slicez=tf.cast(slicez, tf.float32)
#                slicez=tf.expand_dims(slicez,2)
#                slicez=tf.expand_dims(slicez,0)
#                ABsr=generatorSR(slicez)
#                ABsr=(ABsr+1)*127.5
#                ABsr=tf.math.round(ABsr)
#                ABsr=np.asarray(ABsr,'uint8')
#                #image_path = f'{testFile}-SRxy-{z}-{args.scale}x-{args.modelName}.tif'
#                #tifffile.imwrite(image_path, np.squeeze(ABsr))

#                domainSRxy[:,:,z]=ABsr[0,:,:,0]
#                
#            for x in range(domainSRxy.shape[0]):
#                print(f'YZ Pass: Super Resolving slice {x}')
#                slicex=domainSRxy[x,:,:]
#                slicex=slicex/127.5 - 1
#                slicex=tf.transpose(slicex,[1,0])
#                slicex=tf.expand_dims(slicex,2)
#                slicex=tf.expand_dims(slicex,0)
#                ABsr=generatorSRC(slicex)
#                ABsr=np.squeeze(ABsr)
#                ABsr=tf.transpose(ABsr,[1,0])
#                ABsr=(ABsr+1)*127.5
#                domainSRxyz[x,:,:]=np.asarray(ABsr,'uint8')
            # dont transpose, just sqeueze the yz slices and save as png
            domainSRxyz=np.transpose(domainSRxyz,[2,0,1])
            image_path = f'{testFile}-SRxyz-{args.scale}x-{args.modelName}.tif'
            tifffile.imwrite(image_path, np.array(domainSRxyz))
#            for x in range(domainSRxyz.shape[2]):
#                print(f'Saving: slice {x}')
#                tifffile.imwrite(f'{testFile}-SRxyz-{args.scale}x-{args.modelName}_{x}.tif', domainSRxyz[:,:,x])
                
    elif args.phase == 'test':
        # test within scope?
        # read entire LR block of size x,y,zb and upscale to xs,ys,zs

        testFiles = sorted(glob(args.test_dir+'/*.npy'))          

        for testFile in testFiles:
            #testFile=testFiles[0]
            print(f'XY Pass: Super Resolving {testFile}')
            testFileName=testFile.split('/')[-1]
            domain=np.load(testFile, mmap_mode='r')
            domain=np.transpose(domain,[2,0,1])
            domain=domain[0:domain.shape[0]//8*8,0:domain.shape[1]//8*8,0:domain.shape[2]//8*8] 
#            domainSRxy=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]],'uint8')
#            domainSRxyz=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]*args.scale],'uint8')
#            
            
            domainSRxy = h5py.File(f'/{args.test_save_dir}/{testFileName}-SRxy-{args.scale}x-{args.modelName}.h5','w').create_dataset('SRxy', (domain.shape[0]*args.scale, domain.shape[1]*args.scale, domain.shape[2]), dtype='uint8', chunks=(domain.shape[0]*args.scale, domain.shape[1]*args.scale, 1))
##            
#            domainSRxyz = h5py.File(f'/{args.test_save_dir}/{testFileName}-SRxyz-{args.scale}x-{args.modelName}.h5','w').create_dataset('SRxyz', (domain.shape[0]*args.scale, domain.shape[1]*args.scale, domain.shape[2]*args.scale), dtype='uint8', chunks=(1, domain.shape[1]*args.scale, domain.shape[2]*args.scale))
#            
            

            for i in range(domain.shape[2]):
                slicez=domain[:,:,i]
                slicez = (slicez/127.5) - 1 # block will auto cast to float, thanks python
                slicez=tf.cast(slicez, tf.float32)
                slicez=tf.expand_dims(slicez,2)
                slicez=tf.expand_dims(slicez,0)
                maxNz=550
                dualLength=100
                numParts=slicez.shape[2]//(maxNz-dualLength)+1
                z=0
                zz=0
                maxSR=np.zeros([slicez.shape[1]*args.scale,slicez.shape[2]*args.scale],'uint8')
                if numParts==0:
                    print(f'XY Pass: Super Resolving slice {i}')
                    ABsr=generatorSR(slicez)
                    ABsr=np.asarray(ABsr)
                    ABsr=np.squeeze(ABsr)
                    ABsr=(ABsr+1)*127.5
                    ABsr=tf.math.round(ABsr)
                    ABsr=np.asarray(ABsr,'uint8')
                    maxSR=ABsr
                else:
                    for n in range(numParts):
                        print(f'XY Pass: Super Resolving Slice {i} Subsection {n+1}')
                        tempSlice=slicez[:,:,zz:zz+maxNz]
                        ABsr=generatorSR(tempSlice)
                        ABsr=np.asarray(ABsr)
                        ABsr=np.squeeze(ABsr)
                        ABsr=(ABsr+1)*127.5
                        ABsr=tf.math.round(ABsr)
                        ABsr=np.asarray(ABsr,'uint8')
                        if n==0:
                            maxSR[:,:(z+maxNz-dualLength//2)*args.scale]=ABsr[:,:(maxNz-dualLength//2)*args.scale]
                            z=z+maxNz-dualLength//2
                        elif n==numParts-1:
                            maxSR[:,(z)*args.scale:]=ABsr[:,dualLength//2*args.scale:]
                        else:
                            maxSR[:,(z)*args.scale:(z+maxNz-dualLength)*args.scale]=ABsr[:,dualLength//2*args.scale:(maxNz-dualLength//2)*args.scale]
                            z=z+maxNz-dualLength
                        zz=zz+maxNz-dualLength
                domainSRxy[:,:,i]=maxSR
#                np.save(f'{args.test_temp_save_dir}/{testFileName}-SRxy-{i}-{args.scale}x-{args.modelName}.npy', np.squeeze(maxSR))
                if np.mod(i,100)==0:
                    image_path = f'{args.test_temp_save_dir}/{testFileName}-SRxy-{i}-{args.scale}x-{args.modelName}.tif'
                    tifffile.imwrite(image_path, np.squeeze(maxSR))
#                
            for x in range(domain.shape[0]*args.scale):
                maxNz=550
                dualLength=100
                slicex=np.zeros([domain.shape[1]*args.scale,domain.shape[2]],'uint8')
                
#                for i in range(domain.shape[2]):
#                    stick=np.load(f'{args.test_temp_save_dir}/{testFileName}-SRxy-{i}-{args.scale}x-{args.modelName}.npy', mmap_mode='r')
#                    slicex[:,i]=stick[x,:]
                slicex=domainSRxy[x,:,:]
                
                slicex=slicex/127.5 - 1
                slicex=tf.transpose(slicex,[1,0])
                slicex=tf.expand_dims(slicex,2)
                slicex=tf.expand_dims(slicex,0)
                
                
                numParts=slicex.shape[1]//(maxNz-dualLength)+1
                z=0
                zz=0
                maxSR=np.zeros([slicex.shape[1]*args.scale,slicex.shape[2]],'uint8')
                if numParts==0:
                    print(f'YZ Pass: Super Resolving slice {x}')

                    ABsr=generatorSRC(slicex)
                    ABsr=np.squeeze(ABsr)
                    ABsr=(ABsr+1)*127.5
                    ABsr=np.asarray(ABsr,'uint8')
                    maxSR=ABsr
                else:
                    for n in range(numParts):
                        print(f'YZ Pass: Super Resolving Slice {x} Subsection {n+1}')
                        tempSlice=slicex[:,zz:zz+maxNz,:]
                        ABsr=generatorSRC(tempSlice)
                        ABsr=np.asarray(ABsr)
                        ABsr=np.squeeze(ABsr)
                        ABsr=(ABsr+1)*127.5
                        ABsr=tf.math.round(ABsr)
                        ABsr=np.asarray(ABsr,'uint8')
                        if n==0:
                            maxSR[:(z+maxNz-dualLength//2)*args.scale,:]=ABsr[:(maxNz-dualLength//2)*args.scale,:]
                            z=z+maxNz-dualLength//2
                        elif n==numParts-1:
                            maxSR[(z)*args.scale:,:]=ABsr[dualLength//2*args.scale:,:]
                        else:
                            maxSR[(z)*args.scale:(z+maxNz-dualLength)*args.scale,:]=ABsr[dualLength//2*args.scale:(maxNz-dualLength//2)*args.scale,:]
                            z=z+maxNz-dualLength
                        zz=zz+maxNz-dualLength
                maxSR=tf.transpose(maxSR,[1,0])
                
#                np.save(f'{args.test_temp_save_dir}/{testFileName}-SRxyz-{x}-{args.scale}x-{args.modelName}.npy', np.squeeze(maxSR))
                tifffile.imwrite(f'{args.test_save_dir}/{testFileName}-SRxyz-{x}-{args.scale}x-{args.modelName}.tif', np.squeeze(maxSR))
#                domainSRxyz[x,:,:]=np.asarray(maxSR,'uint8')
#                if np.mod(x,100)==0:
#                    image_path = f'{args.test_temp_save_dir}/{testFileName}-SRxyz-{x}-{args.scale}x-{args.modelName}.tif'
#                    tifffile.imwrite(image_path, np.squeeze(maxSR))

            
#            for z in range(domain.shape[2]*args.scale):
#                print(f'Reconstituting z slice {z} into tif')
#                tifffile.imwrite(f'{args.test_save_dir}/{testFileName}-SRxyz-{z}-{args.scale}x-{args.modelName}.tif', domainSRxyz[:,:,z])

#                

