import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib as mpl
import os, sys
import math
import random

from datetime import datetime

#plt.style.use('bmh')
mpl.rcParams['axes.linewidth'] = 1.2 #set the value globally
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Optima'

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['xtick.color'] = 'Grey'
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['ytick.color'] = 'Grey'
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['hatch.color'] = 'Red'
mpl.rcParams['hatch.linewidth'] = 1.5

mpl.rcParams['grid.color'] = 'gainsboro'

mpl.rc('axes',edgecolor='darkgrey')
# plt.rc('font', family='Arail')
# font = {'family' : 'Arail',
#         'weight' : 'normal',
#         'color'  : 'black',
#         'size'   : '12'}

plt.rc('pdf', fonttype=42)
# plt.rc('font', family='Arial', size=10)

colors = ['blue', 'DeepSkyblue', 'Grey', 'Red', 'Orange', 'Tomato', 'LightGrey', 'Black', 'Palegreen', 'Azure']
colors = ['tab:blue', 'tab:orange', 'Grey', 'Red', 'Orange', 'Tomato', 'LightGrey', 'Black', 'Palegreen', 'Azure']
colors = ['blue', 'tab:red', 'Grey', 'Red', 'Orange', 'Tomato', 'LightGrey', 'Black', 'Palegreen', 'Azure']

class Tuple:
    def __init__(self, id, label):
        self.id = id
        self.label = label

    def get_id(self):
        return self.id

    def get_label(self):
        return self.label

class Table:
    def __init__(self, tuple_num, label_1_ratio):
        self.index_list = []
        self.tuple_num = tuple_num
        self.tuple_list = []

        label_1_num = tuple_num * label_1_ratio
        label__1_num = tuple_num - label_1_num

        for i in range(0, tuple_num):
            label = -1
            if i >= label__1_num:
                label = 1
            tuple = Tuple(i, label)
            self.tuple_list.append(tuple)

            self.index_list.append(i)

    def get_index_list(self):
        return self.index_list


    def perform_no_shuffle(self):
        return self.tuple_list

    def perform_fully_shuffle(self):
        random.shuffle(self.tuple_list)
        return self.tuple_list

    def perform_sliding_window_shuffle(self):
        original_list = self.tuple_list
        total_size = len(original_list)
        window_size = int(0.1 * total_size)
        window = []
        new_list = []

        for i in range(0, window_size):
            window.append(original_list[i])

        for i in range(window_size, total_size):
            index = random.randint(0, window_size - 1)
            new_list.append(window[index])
            window[index] = original_list[i]
        
        random.shuffle(window)
        for t in window:
            new_list.append(t)
        assert(len(new_list) == total_size)
        self.tuple_list = new_list

        return self.tuple_list
    
    def perform_mrs_shuffle(self):
        original_list = self.tuple_list
        total_size = len(original_list)

        window_size = int(0.1 * total_size)
        window = []
        new_list = []

        sample_list = random.sample(original_list, window_size)
        final_list = []

        for i in range(0, window_size):
            window.append(original_list[i])
        
        for i in range(window_size, total_size):
            index = random.randint(0, window_size - 1)
            new_list.append(window[index])
            window[index] = original_list[i]
        
        for t in window:
            new_list.append(t)

        sample_index = 0
        new_list_index = 0
        for i in range(0, total_size):
            if random.random() <= 0.35:
                final_list.append(sample_list[sample_index])
                sample_index =  (sample_index + 1) % window_size
            else:
                final_list.append(new_list[new_list_index])
                new_list_index += 1

        
        self.tuple_list = final_list
           
        return self.tuple_list

    def perform_only_page_shuffle(self, block_tuple_num):
        shuffled_tuple_list = []

        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        block_index_list = []

        block_num = int(self.tuple_num / block_tuple_num)
        for i in range(0, block_num):
            block_index_list.append(i)
        
        # [8, 3, 5, 2, 0, 9, 1, 4, 6, 7]
        random.shuffle(block_index_list)


        for i in range(0, block_num):
            for j in range(0, block_tuple_num):
                index = block_index_list[i] * block_tuple_num + j
                tuple = self.tuple_list[index]
                shuffled_tuple_list.append(tuple)
        
        return shuffled_tuple_list
    
    def perform_page_tuple_shuffle(self, block_tuple_num, buffer_tuple_num):
        page_shuffled_tuple_list = self.perform_only_page_shuffle(block_tuple_num)
        buffer = []

        page_tuple_shuffled_list = []

        for i in range(0, self.tuple_num):
            tuple = page_shuffled_tuple_list[i]

            buffer.append(tuple)

            if len(buffer) == buffer_tuple_num:
                random.shuffle(buffer)
                for t in buffer:
                    page_tuple_shuffled_list.append(t)
                buffer.clear()
            
        if buffer:
            for t in buffer:
                page_tuple_shuffled_list.append(t)
        
        return page_tuple_shuffled_list



def plot_id_distribution(shuffle_mode, tuple_num, label_1_ratio, block_tuple_num, buffer_tuple_num, title, outputFile):

    table = Table(tuple_num, label_1_ratio)
    tuple_list = []
    index_list = table.get_index_list()

    if shuffle_mode == 'no_shuffle':
        tuple_list = table.perform_no_shuffle()
    if shuffle_mode == 'fully_shuffle':
        tuple_list = table.perform_fully_shuffle()
    elif shuffle_mode == 'only_page_shuffle':
        tuple_list = table.perform_only_page_shuffle(block_tuple_num)
    elif shuffle_mode == 'page_tuple_shuffle':
        tuple_list = table.perform_page_tuple_shuffle(block_tuple_num, buffer_tuple_num)
    elif shuffle_mode == 'sliding_window_shuffle':
        tuple_list = table.perform_sliding_window_shuffle()
    elif shuffle_mode == 'mrs_shuffle':
        tuple_list = table.perform_mrs_shuffle()


  
    postive_id_list = []
    positive_index_list = []

    negative_id_list = []
    negative_index_list = []

    for i in range(0, len(tuple_list)):
        tuple = tuple_list[i]
        id = tuple.get_id()
        label = tuple.get_label()

        if (label == 1):
            postive_id_list.append(id)
            positive_index_list.append(i)
        else:
            negative_id_list.append(id)
            negative_index_list.append(i)

    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(111)

    plt.subplots_adjust(left=0.198, bottom=0.17, right=0.946, top=0.967,
                 wspace=0.205, hspace=0.2)   
    # plt.subplots_adjust(left=0.16, bottom=0.11, right=0.94, top=0.88,
    #             wspace=0.2, hspace=0.2)   
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey=False, sharex= True, figsize=(4,4.4))
    # plt.subplots_adjust(wspace=0, hspace=0)
    y_label = "Tuple id"
    ax.set_ylabel(y_label, color='black')
    ax.set_xlabel("The i-th tuple")

    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.scatter(negative_index_list, negative_id_list, s=1, color=colors[0])
    ax.scatter(positive_index_list, postive_id_list, s=1, color=colors[0])

    ax.grid(True)
    # ax.set_ylim(ymax=2500)
    # ax.set_xlim(xmax=2500)
    #ax.set_ylim(ymax=2000)
    ax.set_xlim(xmin=0, xmax=1000)
    ax.set_ylim(ymin=0, ymax=1000)
    #ax.set_xlim(xmax=12500)
   
    #ax.legend(loc='upper right')
    
    #plt.suptitle(title, y=0.95)
    #plt.suptitle(title)

    fig = plt.gcf()
    plt.show()
    # fig.savefig(outputFile, dpi=300, bbox_inches='tight')


def plot_label_distribution(shuffle_mode, tuple_num, label_1_ratio, block_tuple_num, buffer_tuple_num, title, outputFile):

    table = Table(tuple_num, label_1_ratio)
    tuple_list = []
    index_list = table.get_index_list()

    if shuffle_mode == 'no_shuffle':
        tuple_list = table.perform_no_shuffle()
    if shuffle_mode == 'fully_shuffle':
        tuple_list = table.perform_fully_shuffle()
    elif shuffle_mode == 'only_page_shuffle':
        tuple_list = table.perform_only_page_shuffle(block_tuple_num)
    elif shuffle_mode == 'page_tuple_shuffle':
        tuple_list = table.perform_page_tuple_shuffle(block_tuple_num, buffer_tuple_num)
    elif shuffle_mode == 'sliding_window_shuffle':
        tuple_list = table.perform_sliding_window_shuffle()
    elif shuffle_mode == 'mrs_shuffle':
        tuple_list = table.perform_mrs_shuffle()

    
    fig = plt.figure(figsize=(3.2, 2.8))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.157, bottom=0.186, right=0.964, top=0.976,
                 wspace=0.205, hspace=0.2)   
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey=False, sharex= True, figsize=(4,4.4))
    # plt.subplots_adjust(wspace=0, hspace=0)
    y_label = "#tuples"
    ax.set_ylabel(y_label, color='black')
    ax.set_xlabel("The i-th batch (20 tuples per batch)")
   

    #label_list = table.get_column("label")

    batch_index_list = []
    label0_list = []
    label1_list = []

    label1_count = 0
    label0_count = 0
    batch_size = 20


    for i in range(len(index_list)):
        label = tuple_list[i].get_label()
        if label == 1:
            label1_count += 1
        else:
            label0_count += 1

        if (i + 1) % batch_size == 0 or i == len(index_list) - 1:
            batch_index_list.append((i + 1)/ batch_size)
            label1_list.append(label1_count)
            label0_list.append(label0_count)
            label1_count = 0
            label0_count = 0
    

    ax.plot(batch_index_list, label1_list, label = 'label=+1', color=colors[0])
    ax.plot(batch_index_list, label0_list, label = 'label=-1', color=colors[1], linestyle='--')

    #ax.grid(True)
    ax.set_ylim(ymin=-1)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=batch_size + 1)
    #ax.set_xlim(xmax=12500)
   
    ax.legend()
    
    #plt.suptitle(title + ' (every ' + str(batch_size) + ' tuples)', y=0.95)

    fig = plt.gcf()
    plt.show()
    #fig.savefig(outputFile, dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    tuple_num = 1000
    block_tuple_num = 25
    buffer_tuple_num = 250
    label_1_ratio = 0.5
    
    # shuffle_mode = 'no_shuffle'
    # title = 'No shuffle'

    # shuffle_mode = 'fully_shuffle'
    # title = 'Fully shuffle'

    # shuffle_mode = 'only_page_shuffle'
    # title = 'Only block shuffle'

    #
    # if shuffle_mode == 'no_shuffle':
    #     tuple_list = table.perform_no_shuffle()
    # if shuffle_mode == 'fully_shuffle':
    #     tuple_list = table.perform_fully_shuffle()
    # elif shuffle_mode == 'only_page_shuffle':
    #     tuple_list = table.perform_only_page_shuffle(block_tuple_num)
    # elif shuffle_mode == 'page_tuple_shuffle':
    #     tuple_list = table.perform_page_tuple_shuffle(block_tuple_num, buffer_tuple_num)
    # elif shuffle_mode == 'sliding_window_shuffle':
    #     tuple_list = table.perform_sliding_window_shuffle()
    # elif shuffle_mode == 'mrs_shuffle':
    #     tuple_list = table.perform_mrs_shuffle()


    shuffle_mode = 'page_tuple_shuffle'
    title = 'Block+tuple shuffle'

    # shuffle_mode = 'sliding_window_shuffle'
    # title = 'Sliding window shuffle'

    # shuffle_mode = 'mrs_shuffle'
    # title = 'MRS shuffle'


    outputFile = './picture'

    plot_id_distribution(shuffle_mode, tuple_num, label_1_ratio, block_tuple_num, buffer_tuple_num, title, outputFile)
    plot_label_distribution(shuffle_mode, tuple_num, label_1_ratio, block_tuple_num, buffer_tuple_num, title, outputFile)
    # plot_label_scatter(input_dir, data_table, title, outputFile + "-label.png")
    #plot_label_distribution(shuffle_mode, title, outputFile + "-label-lines.png")




'''
def plot_id_curve(input_dir, data_table,  title, outputFile) :

    file_to_plot = input_dir + data_table + ".txt"


    table = Table()
    table.init(file_to_plot)
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.16, bottom=0.11, right=0.94, top=0.88,
                 wspace=0.2, hspace=0.2)   
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey=False, sharex= True, figsize=(4,4.4))
    # plt.subplots_adjust(wspace=0, hspace=0)
    y_label = "id"
    ax.set_ylabel(y_label, color='black')
    ax.set_xlabel("id")
   

    id_list = table.get_column("id")
    index_list = table.get_column("index")
    
    ax.scatter(index_list, id_list, s=0.1)

    ax.grid(True)
    # ax.set_ylim(ymax=2500)
    # ax.set_xlim(xmax=2500)
    #ax.set_ylim(ymax=2000)
    # ax.set_xlim(xmin=4000)
    # ax.set_ylim(ymax=8000)
    #ax.set_xlim(xmax=12500)
   
    #ax.legend(loc='upper right')
    
    plt.suptitle(title, y=0.95)

    fig = plt.gcf()
    #plt.show()
    fig.savefig(outputFile, dpi=300, bbox_inches='tight')
'''
'''
def plot_label_scatter(input_dir, data_table,  title, outputFile) :

    file_to_plot = input_dir + data_table + ".txt"


    table = Table()
    table.init(file_to_plot)
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.16, bottom=0.11, right=0.94, top=0.88,
                 wspace=0.2, hspace=0.2)   
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey=False, sharex= True, figsize=(4,4.4))
    # plt.subplots_adjust(wspace=0, hspace=0)
    y_label = "label"
    ax.set_ylabel(y_label, color='black')
    ax.set_xlabel("id")
   

    label_list = table.get_column("label")
    index_list = table.get_column("index")
    
    ax.scatter(index_list, label_list, s=0.1, color='r')

    #ax.plot(index_list, label_list, ms=0.1, color='r')

    #ax.grid(True)
    #ax.set_ylim(ymin=4000)
    #ax.set_xlim(xmin=4000)
    #ax.set_ylim(ymax=8000)
    #ax.set_xlim(xmax=12500)
   
    #ax.legend(loc='upper right')
    
    plt.suptitle(title, y=0.95)

    fig = plt.gcf()
    #plt.show()
    fig.savefig(outputFile, dpi=300, bbox_inches='tight')

def plot_label_lines(input_dir, data_table,  title, outputFile) :

    file_to_plot = input_dir + data_table + ".txt"

    table = Table()
    table.init(file_to_plot)
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.16, bottom=0.11, right=0.94, top=0.88,
                 wspace=0.2, hspace=0.2)   
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey=False, sharex= True, figsize=(4,4.4))
    # plt.subplots_adjust(wspace=0, hspace=0)
    y_label = "label"
    ax.set_ylabel(y_label, color='black')
    ax.set_xlabel("id")
   

    label_list = table.get_column("label")
    index_list = table.get_column("index")

    batch_index_list = []
    label0_list = []
    label1_list = []

    label1_count = 0
    label0_count = 0
    batch_size = 100
    for i in range(len(index_list)):
        if label_list[i] == 1:
            label1_count += 1
        else:
            label0_count += 1

        if (i + 1) % batch_size == 0 or i == len(index_list) - 1:
            batch_index_list.append(i)
            label1_list.append(label1_count)
            label0_list.append(label0_count)
            label1_count = 0
            label0_count = 0
    

    ax.plot(batch_index_list, label1_list, label = 'label=1')
    ax.plot(batch_index_list, label0_list, label = 'label=-1')

    #ax.grid(True)
    ax.set_ylim(ymin=0)
    #ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=batch_size)
    #ax.set_xlim(xmax=12500)
   
    ax.legend(loc='upper right')
    
    plt.suptitle(title + ' (every ' + str(batch_size) + ' tuples)', y=0.95)

    fig = plt.gcf()
    #plt.show()
    fig.savefig(outputFile, dpi=300, bbox_inches='tight')
'''