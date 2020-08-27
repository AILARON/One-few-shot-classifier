import config
import main
import time
import matplotlib.pyplot as plt
from os import walk
import os


def write_log(f,run_nr, acc, un_acc, class_acc, val_time):
    if config.error_message != '':
        f.write(("Error message: " + config.error_message))
    f.write(("Network: " + config.active_model_name+ '\n'))
    f.write(("Run nr: "+ str(run_nr) + '\n'))
    f.write(("Classification type: " + config.classification_type+ '\n'))
    f.write(("Dataset: " + config.dataset + '\n'))
    f.write(("Testing for unknown species: " + str(config.testing_for_unknwon_species)+ '\n'))
    f.write(("K-way: " + str(config.k_way)+ '\n'))
    f.write(("N-shot: " + str(config.n_shot)+ '\n'))
    f.write("Learning rate: "+ str(config.learning_rate) + '\n')
    f.write("Weight decay: " + str(config.weight_decay) + '\n')
    f.write("Epochs: " + str(config.epochs) + '\n')
    f.write("Batch size: " + str(config.batch_size) + '\n')
    f.write("Image size: " + str(config.image_size) + 'x' + str(config.image_size) + '\n')
    f.write("Training samples: " + str(config.training_examples) + '\n')
    f.write("Validation samples: " + str(config.validation_examples) + '\n')
    f.write("Data split: " + str(config.train_percent) + "/" + str(config.val_percent) + '/' + str(config.test_percent) + '\n')
    f.write("Image computation speed: " + str(val_time) + '\n')
    f.write("--------------------------------------------\n")
    f.write("Accuracy: " + str(acc) + '\n')
    if config.testing_for_unknwon_species == True:
        f.write("Unknown class Accuracy: " + str(un_acc) + '\n')
        f.write("Classificaiton Accuracy: " + str(class_acc) + '\n')
    f.write("--------------------------------------------\n")


def save_log(filename, run_nr, acc, un_acc, class_acc, val_time):
    f = open(filename, "a")
    write_log(f,run_nr, acc, un_acc, class_acc, val_time)
    f.close()

def check_highscore(filename="highscore.txt"):
    if os.path.isfile(filename):
        f = open(filename, "r")
        for line in f:
            if "Accuracy" in line:
                highscore_acc = float(line[10:13])
        f.close()
        return highscore_acc
    else:
        print('Creating ' + filename)
        return 0

def write_highscore(f, run_nr, acc):
    f.write(("Run nr: "+ str(run_nr) + '\n'))
    f.write(("Model: "+ str(config.active_model_name) + '\n'))
    f.write("Accuracy: " + str(acc) + '\n')

def save_highscore(run_nr, acc, filename="highscore.txt"):
    f = open(filename, "w")
    write_highscore(f, run_nr, acc)
    f.close()

def get_runNr():
    latest_run = 0
    for (dirpath, dirnames, filenames) in walk('Results', topdown=True):
        for graph_name in filenames:
            graph_name = graph_name[:-4]
            underscore_index = graph_name.find('_')
            if int(graph_name[underscore_index+1:]) > latest_run:
                latest_run = int(graph_name[underscore_index+1:])
    return latest_run + 1

def get_validation_timestamp(N_batches):
    timestamps = []
    batch_per_epoch = int(N_batches / config.epochs)
    for i in range(config.epochs):
        timestamps.append(i*batch_per_epoch)
    return timestamps

def get_classification_type():
    while True:
        print('Insert classification type: \n 1) Traditional classification\n 2) One/Few-shot classification\n 3) New class classification\n [1/2/3]')
        clasif_type = int(input(':'))
        if clasif_type == 1:
            config.classification_type = 'traditional'
            return
        elif clasif_type == 2:
            config.classification_type = 'one_few_shot'
            return
        elif clasif_type == 3:
            config.classification_type = 'one_few_shot'
            config.testing_for_unknwon_species = True
            return
        else:
            print('ERROR: Invalid input, try again')


filename = "Log.txt"
run_nr = get_runNr()
start_time = time.time()
get_classification_type()


for model in config.models:
    config.active_model_name = model
    print('Run nr: ' + str(run_nr))
    print("Model: ", model)
    acc, un_acc, class_acc, loss_record, val_time = main.one_shot_classifier()
    save_log(filename, run_nr, acc, un_acc, class_acc, val_time)

    highscore_acc = check_highscore()
    if highscore_acc < acc:
        print('highscore_acc: ' + str(highscore_acc))
        print('New highscore: ' + str(acc))
        save_highscore(run_nr, acc)

    #validation_timestamps = get_validation_timestamp(len(loss_record))
    plt.plot(loss_record)
    #plt.plot(x=validation_timestamps, y=val_loss_record)
    #plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plotName = config.dataset + '_' + str(run_nr)
    plt.savefig(config.plot_save_file + '/' + plotName+'.png')
    #plt.show()
    plt.clf()
    config.error_message = ''
    print("Time: %s minutes"%((time.time()-start_time)/60))

    run_nr += 1
