# Main code for training ERFNet model in Mapillary Dataset
# Sept 2018
# Eduardo Romera, Kailun Yang
#######################

import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps, ImageEnhance
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Pad, Resize
from torchvision.transforms import ToTensor, ToPILImage

from piwise.dataset import VOC12,cityscapes
from piwise.criterion import CrossEntropyLoss2d,FocalLoss2d
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard
from piwise.ModelDataParallel import ModelDataParallel,CriterionDataParallel #https://github.com/pytorch/pytorch/issues/1893

import importlib
import evalIoU

from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 28

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()
input_transform = Compose([
    CenterCrop(240),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    CenterCrop(240),
    ToLabel(),
    Relabel(255, 27),
])

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=480):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Scale(self.height, Image.BILINEAR)(input)
        target = Scale(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        
            degree=random.randint(-20,20)
            input=input.rotate(degree,resample=Image.BILINEAR,expand=True)
            target=target.rotate(degree,resample=Image.NEAREST,expand=True)

            w, h=input.size
            nratio=random.uniform(0.5,1.0) 
            ni=random.randint(0,int(h-nratio*h))
            nj=random.randint(0,int(w-nratio*w))
            input=input.crop((nj,ni,int(nj+nratio*w),int(ni+nratio*h)))
            target=target.crop((nj,ni,int(nj+nratio*w),int(ni+nratio*h)))
            input=Resize((480,640),Image.BILINEAR)(input)
            target=Resize((480,640),Image.NEAREST)(target)      

            brightness_factor=random.uniform(0.8,1.2)
            contrast_factor=random.uniform(0.8,1.2)
            saturation_factor=random.uniform(0.8,1.2)
            #sharpness_factor=random.uniform(0.0,2.0)
            hue_factor=random.uniform(-0.2,0.2)

            enhancer1=ImageEnhance.Brightness(input)
            input=enhancer1.enhance(brightness_factor)

            enhancer2=ImageEnhance.Contrast(input)
            input=enhancer2.enhance(contrast_factor)

            enhancer3=ImageEnhance.Color(input)
            input=enhancer3.enhance(saturation_factor)

            #enhancer4=ImageEnhance.Sharpness(input)
            #input=enhancer4.enhance(sharpness_factor)

            input_mode=input.mode
            h, s, v=input.convert('HSV').split()
            np_h=np.array(h,dtype=np.uint8)
            with np.errstate(over='ignore'):
                np_h+=np.uint8(hue_factor*255)
            h=Image.fromarray(np_h,'L')
            input=Image.merge('HSV',(h,s,v)).convert(input_mode) 

        else:
            input=Resize((480,640),Image.BILINEAR)(input)
            target=Resize((480,640),Image.NEAREST)(target)
            
        input = ToTensor()(input)
       
        if (self.enc):
            target = Resize((60,80), Image.NEAREST)(target)
        target = ToLabel()(target)
        target=  Relabel(255, 27)(target)

        return input, target

best_acc = 0

def train(args, model, enc=False):
    global best_acc

    weight = torch.ones(NUM_CLASSES)
    weight[0] = 121.21	
    weight[1] = 947.02	
    weight[2] = 151.92	
    weight[3] = 428.31	
    weight[4] = 25.88	
    weight[5] = 235.97	
    weight[6] = 885.72	
    weight[7] = 911.87	
    weight[8] = 307.49	
    weight[9] = 204.69
    weight[10]= 813.92	
    weight[11]= 5.83	
    weight[12]= 34.22	
    weight[13]= 453.34	
    weight[14]= 346.10	
    weight[15]= 250.19	
    weight[16]= 119.99	
    weight[17]= 75.28	
    weight[18]= 76.71
    weight[19]= 8.58
    weight[20]= 281.68
    weight[21]= 924.07
    weight[22]= 3.91
    weight[23]= 7.14
    weight[24]= 88.89
    weight[25]= 59.00
    weight[26]= 126.59
    weight[27]=  0

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(enc, augment=True, height=args.height)#1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)#1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        #criterion = CrossEntropyLoss2d(weight.cuda()) 
        criterion=FocalLoss2d(weight.cuda()) 
    else:
        #criterion = CrossEntropyLoss2d(weight)
        criterion = FocalLoss2d(weight.cuda()) 

    print(type(criterion))

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))
    
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-5, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)      ## scheduler 2

    start_epoch = 1

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        #TODO: remake the evalIoU.py code to avoid using "evalIoU.args"
        confMatrix    = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader): 
            start_time = time.time()   
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs, only_encode=enc)  
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            #loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            time_train.append(time.time() - start_time)
        
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)   
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        #evalIoU.printConfMatrix(confMatrix, evalIoU.args)

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        #New confusion matrix data
        confMatrix    = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            outputs = model(inputs, only_encode=enc) 

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)


            #Add outputs to confusion matrix
            if (doIouVal):
                #compatibility with criterion dataparallel
                if isinstance(outputs, list):   #merge gpu tensors
                    outputs_cpu = outputs[0].cpu()
                    for i in range(1,len(outputs)):
                        outputs_cpu = torch.cat((outputs_cpu, outputs[i].cpu()), 0)
                    #print(outputs_cpu.size())
                else:
                    outputs_cpu = outputs.cpu()

                #start_time_iou = time.time()
                for i in range(0, outputs_cpu.size(0)):   #args.batch_size
                    prediction = ToPILImage()(outputs_cpu[i].max(0)[1].data.unsqueeze(0).byte())
                    groundtruth = ToPILImage()(labels[i].cpu().byte())
                    nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth, confMatrix, perImageStats, evalIoU.args)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        # Calculate IOU scores on class level from matrix
        iouVal = 0
        iouTrain=0
        if (doIouVal):
            #start_time_iou = time.time()
            classScoreList = {}
            for label in evalIoU.args.evalLabels:
                labelName = evalIoU.trainId2label[label].name
                classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)

            iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
            iouVal = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
            print ("EPOCH IoU on VAL set: ", iouAvgStr)
           
        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if (enc and epoch == args.num_epochs):
            best_acc=0          

        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth'
            filenameBest = savedir + '/model_best_enc.pth'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth'
            filenameBest = savedir + '/model_best.pth'
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best_each.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best_each.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        #if (True) #(is_best):
        torch.save(model.state_dict(), filenamebest)
        print(f'save: {filenamebest} (epoch: {epoch})')
        filenameSuperBest=f'{savedir}/model_superbest.pth'
        if(is_best):
            torch.save(model.state_dict(),filenameSuperBest)
            print(f'saving superbest') 
        if (not enc):
            with open(savedir + "/best.txt", "w") as myfile:
                 myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
        else:
            with open(savedir + "/best_encoder.txt", "w") as myfile:
                 myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        #model = ModelDataParallel(model).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
   
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
        else:
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet_pspnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes a lot to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True) #calculating IoU takes about 0,10 seconds per image ~ 50s per 500 images in VAL set, so 50 extra secs per epoch    
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
