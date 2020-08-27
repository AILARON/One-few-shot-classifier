import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data_loader
import model
import time
import config

def initWeights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.2)
        m.bias.data.normal_(mean = 0.5, std = 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias != None:
            m.bias.data.normal_(mean = 0.5, std = 0.01)

def generateLabels(batch_size):
    labels = torch.zeros(batch_size)
    for i in range(batch_size):
        if i%2==0:
            labels[i]=1
    return labels

def validate(net,img1,img2,val_labels,criterion):
    with torch.no_grad():
        net.eval()
        output = net.forward(img1,img2)
        loss = criterion(output, val_labels)
        current_loss = loss.item()
        return current_loss


def getModel():
    if config.active_model_name == 'default':
        net = model.SiameseNet()
    elif config.active_model_name[:3] == 'vgg':
        net = model.SiameseVGGNet()
    elif config.active_model_name[:6] == 'resnet':
        net = model.SiameseResNet()
    elif config.active_model_name == 'planktonNet':
        net = model.PlanktonNet()
    elif config.active_model_name == 'customVGG11':
        net = model.SiameseCustomVGG11Net()
    elif config.active_model_name == 'customResNet18':
        net = model.SiameseCustomRes18Net()
    elif config.active_model_name == 'customRes2Net18':
        net = model.SiameseCustom2Res18Net()
    else:
        error = "No " + config.active_model_name + " model found"
        print(error)
        config.error_message += error + '\n'
    return net


def one_shot_classifier():
    start_time = time.time()
    print(generateLabels(128))

    net = getModel()
    if torch.cuda.is_available():
      net.cuda()
    net.apply(initWeights)

    optimizer = optim.Adam(net.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

    criterion = nn.BCELoss()

    if config.active_model_name == 'default':
        n_channels = 1
    else:
        n_channels = 3
    loss_record = []
    if config.load_model == False:
        print("----Loading Data-----")
        if config.classification_type == 'traditional' and config.k_way == 1 and config.n_shot == 1:
            all_images, num_images = data_loader.getImages()
            train_batches = data_loader.getTrainBatches(all_images, num_images, n = config.training_examples, batch_size = config.batch_size)
        elif config.classification_type == 'traditional':
            train_images, num_train_images = data_loader.getOneShotImages('background')
            train_batches = data_loader.getTrainBatches(train_images, num_train_images, n=config.training_examples, batch_size = config.batch_size)
        elif config.classification_type == 'one_few_shot':
            train_images, num_train_images = data_loader.getOneShotImages('background')
            train_batches = data_loader.getTrainBatches(train_images, num_train_images, n=config.training_examples, batch_size = config.batch_size)
        #val_batch = data_loader.getValBatch(all_images, num_images)

        print("Data loaded in %s seconds"%(time.time()-start_time))

        start_time = time.time()
        print("----Training-----")


        for epoch in range(1,config.epochs+1):
            net.train()

            for batch_idx,mini_batch in enumerate(train_batches):

                optimizer.zero_grad()

                batch_size = mini_batch.size()[0]
                img1 = mini_batch[:,0,:,:,:].view(-1,n_channels,config.image_size,config.image_size)
                img2 = mini_batch[:,1,:,:,:].view(-1,n_channels,config.image_size,config.image_size)
                labels = generateLabels(batch_size)
                if torch.cuda.is_available():
                  img1 = img1.cuda().float()
                  img2 = img2.cuda().float()
                  labels = labels.cuda()
                output = net.forward(img1,img2)

                loss = criterion(output, labels)
                current_loss = loss.item()
                loss_record.append(current_loss)
                loss.backward()
                optimizer.step()
                print("Epoch:%d Batch:%d Loss:%.5f Time Lapsed:%s"%(epoch,batch_idx+1,current_loss,time.time() - start_time))
        torch.save(net.state_dict(), config.MODEL_PATH + 'one_shot.pth')
    else:
        net = getModel()
        net.load_state_dict(torch.load(config.MODEL_PATH + 'one_shot.pth'))
        net.eval()


        '''val_labels = generateLabels(config.validation_examples)
        val_img1 = val_batch[:,0,:,:,:].view(-1,3,config.image_size,config.image_size)
        val_img2 = val_batch[:,1,:,:,:].view(-1,3,config.image_size,config.image_size)
        if torch.cuda.is_available():
            net.cuda()
            val_img1 = val_img1.cuda().float()
            val_img2 = val_img2.cuda().float()
            val_labels = val_labels.cuda()
        val_loss_record.append(validate(net,val_img1,val_img2,val_labels,criterion))'''

    #plt.plot(loss_record)
    #plt.show()
    val_loss_record = []
    q_class_nrs = []
    if config.classification_type == 'traditional' and config.k_way == 1 and config.n_shot == 1:
        '''True/false classification'''
        val_batches = data_loader.getValBatch(all_images, num_images)
    elif config.classification_type == 'traditional':
        val_images, num_val_images = data_loader.getOneShotImages('evaluation')
        val_batches, q_class_nrs, val_batches_name = data_loader.getFewShotValBatches(val_images, num_val_images)
    elif config.classification_type == 'one_few_shot':
        val_images, num_val_images = data_loader.getOneShotImages('evaluation')
        if config.k_way == 1 & config.n_shot == 1:
            '''Normal true/false one-shot verificaiton'''
            val_batches = data_loader.getValBatch(val_images, num_val_images)
        #elif config.k_way == 1 & config.n_shot != 1:
            '''True/false n-shot verification'''
        else:
            '''k-way classificaiton'''
            val_batches, q_class_nrs, val_batches_name = data_loader.getFewShotValBatches(val_images, num_val_images)


    p = 0
    correct = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    un_tp, un_fp, un_tn, un_fn = 0, 0 ,0 ,0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    correct_classification = 0
    unknown_class_correct = 0
    val_start_time = time.time()
    for batch_idx,mini_batch in enumerate(val_batches):


        with torch.no_grad() :
            #net = model.SiameseNet()
            #net.load_state_dict(torch.load('siamese.pth'))
            net.eval()
            batch_size = mini_batch.size()[0]
            img1 = mini_batch[:,0,:,:,:].view(-1,n_channels,config.image_size,config.image_size)
            img2 = mini_batch[:,1,:,:,:].view(-1,n_channels,config.image_size,config.image_size)
            validation_start_time = time.time()
            if torch.cuda.is_available():
                net.cuda()
                img1 = img1.cuda().float()
                img2 = img2.cuda().float()
            output = net.forward(img1,img2)

            if config.k_way == 1 or config.classification_type == 'traditional':
                output = output >= 0.5
                for i in range(batch_size):
                    correct_bool = False
                    #print('Prediction: ', output[i])
                    if(i%2 == 0):
                        if(output[i] == 1):
                            correct += 1
                            correct_bool = True
                            true_positives += 1
                        else:
                            false_positives += 1
                    else:
                        if(output[i] == 0):
                            correct += 1
                            correct_bool = True
                            true_negatives += 1
                        else:
                            false_negatives += 1
                    #print('Correct : ', correct_bool)
                    #print('')
            else:
                for l in range(int(batch_size/(config.k_way*config.n_shot))):
                    highest_value = 0
                    best_class_average = 0
                    task_q_class = q_class_nrs[p*config.val_batch_size+l]
                    task_scores = output[l*config.k_way*config.n_shot:(l+1)*config.k_way*config.n_shot]
                    task_score_names = val_batches_name[p][l*config.k_way*config.n_shot:(l+1)*config.k_way*config.n_shot]
                    #print('###########################################################')
                    print('Q_class: ', task_q_class)
                    for i in range(config.k_way):
                        #print('Classnr: ', i)
                        class_average = 0
                        class_scores = task_scores[i*config.n_shot:(i+1)*config.n_shot]
                        class_scores_names = task_score_names[i*config.n_shot:(i+1)*config.n_shot]
                        print('Current class:', i)
                        for j in range(config.n_shot):
                            score = class_scores[j]
                            score_name = class_scores_names[j]
                            class_average += score
                            print('Img1: ', score_name[0][29:])
                            print('Img2: ', score_name[1][29:])
                            print('Score: ', score)
                            if score > 0.5 and task_q_class == i:
                                true_positives += 1
                                print('True positive')
                            elif score > 0.5 and task_q_class != i:
                                false_positives += 1
                                print('False positive')
                            elif score < 0.5 and task_q_class != i:
                                true_negatives += 1
                                print('True negative')
                            elif score < 0.5 and task_q_class == i:
                                false_negatives += 1
                                print('False negative')
                            #print('Score: ', score)
                            #print('-------------------------------------------------')

                            if score > highest_value:
                                highest_value = score
                                highest_value_class = i

                        class_average = class_average/config.n_shot
                        print('Class average: ', class_average)
                        #print('_________________________________________________________________')
                        if class_average > best_class_average:
                            best_class_average = class_average
                            best_scoring_class = i
                    #print('Best scoring class: ', best_scoring_class)


                    tp_bool, tn_bool = False, False
                    if config.testing_for_unknwon_species == True:
                        if task_q_class != -1 and best_class_average >= 0.5:
                            tp_bool = True
                            un_tp += 1
                        elif task_q_class == -1 and best_class_average < 0.5:
                            tn_bool = True
                            un_tn += 1
                        elif task_q_class == -1 and best_class_average >= 0.5:
                            un_fp += 1
                        elif task_q_class != -1 and best_class_average < 0.5:
                            un_fn += 1

                    if (tp_bool and best_scoring_class == task_q_class) or tn_bool:
                        correct2 += 1

                    if (tp_bool and highest_value_class == task_q_class) or tn_bool:
                        correct3 += 1



                    if config.testing_for_unknwon_species == True and best_class_average < 0.5 and task_q_class == -1:
                        unknown_class_correct+=1
                        correct+=1
                    if best_scoring_class == task_q_class:
                        correct_classification+=1
                        correct+=1
                    if highest_value_class == task_q_class:
                        correct4+=1
        p += 1


    un_acc = 0
    class_acc = 0
    val_time = (time.time()-val_start_time)/config.validation_examples
    acc = correct/float(config.validation_examples)
    print('Total acc: ', acc)
    print('Total acc4: ', correct4/float(config.validation_examples))
    print('True positive classifications, one shot: ', true_positives)
    print('False positive classifications, one shot: ', false_positives)
    print('True negatives classifications, one shot: ', true_negatives)
    print('Fase negatives classifications, one shot: ', false_negatives)
    #print(q_class_nrs)
    if q_class_nrs != [] and config.testing_for_unknwon_species == True:
        un_acc = (un_tp + un_tn)/(un_tp + un_tn + un_fp + un_fn) #unknown_class_correct/q_class_nrs.count(-1)
        class_acc = correct_classification/(len(q_class_nrs)-q_class_nrs.count(-1))
        print('True positive rejections: ', un_tp)
        print('False positive rejections: ', un_fp)
        print('True negatives rejections: ', un_tn)
        print('Fase negatives rejections: ', un_fn)
        print('Correct2: ', correct2/config.validation_examples)
        print('Correct3: ', correct3/config.validation_examples)
        print('Unknwon class acc: ', un_acc)
        print('Classification acc: ', class_acc)
    return acc, un_acc, class_acc, loss_record, val_time
