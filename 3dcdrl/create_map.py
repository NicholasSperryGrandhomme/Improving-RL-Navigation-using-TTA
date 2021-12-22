from matplotlib import image
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse


def arguments_given():
    parser = argparse.ArgumentParser()
    p = parser.add_argument
    
    p('--path_txt_pos_track', default="./map_creation/TTA_position/baseline/baseline_original.txt", help='Folder of txt file with positions',  type=str)
    p('--image_path',         default="./scenarios/custom_scenarios/labyrinth/9/test/map.png",      help='The map to draw on',  type=str)
    p('--map_path',           default="./map_creation/map/",                                        help='Path where maps get saved',  type=str)
    p('--arrow_path',         default="./map_creation/arrows/",                                     help='Path where arrows get saved',  type=str)
    p('--view_path',          default="./map_creation/TTA_view/baseline/view/",                     help='Path where the view was saved',  type=str)
    p('--mosaic_path',        default="./map_creation/mosaic/",                                     help='Path where mosaics get saved',  type=str)
    p('--frame_no',           default=30,                                                           help='Numbers of frames which should be used',  type=int)
    p('--jump_steps',         default=4,                                                            help='Jump frames',  type=int)
    p('--map_x_minus',        default=640,                                                          help='SPECIFIC TO MAP 000 FROM LAB 9 TEST, map negative x-axis - find out in SLADE',  type=int)
    p('--map_y_plus',         default=512,                                                          help='SPECIFIC TO MAP 000 FROM LAB 9 TEST, map positive y-axis - find out in SLADE',  type=int)
    
    args = parser.parse_args()
    return args

size_scatter = 10.0

def read_txt_pos_track(txt_path):
    """ Read the position txt file and get the position to output a lists of x and y coordinates."""
    # open / read txt
    f = open(txt_path, "r")
    # load into list
    x_list, y_list = [], []
    if f.mode == "r":
        f_read = f.readlines()
        for l in f_read:
            cord_split = l.split(",")
            x = int(cord_split[0])
            y = int(cord_split[1])
            # image 0,0 cord are at the top left, top left from cord is -640, 512
            x += map_x_minus
            if y >= 0:
                y -= map_y_plus
                y = abs(y)
            else:
                y = abs(y) + map_y_plus
            x_list.append(x) # split return a list
            y_list.append(y)
    return x_list, y_list

def create_track_on_map(txt_path, image_path, steps=False):
    """ Draw position of agent onto the given map. """
    x_list, y_list = read_txt_pos_track(txt_path)
    # to read the image stored in the working directory
    data = image.imread(image_path)
    colormap = cm.rainbow(np.linspace(0, 1, len(x_list[::jump_steps])))
    print(len(colormap))
    plt.imshow(data)
    if steps:
        # save every step
        c = 0
        for x, y in zip(x_list[1::jump_steps], y_list[1::jump_steps]):
            plt.scatter(x, y, s=size_scatter, c=colormap[c])
            plt.axis('off')
            name = "./map/" + "map" + str(c) + ".png"
            plt.savefig(name)
            print("counter", c)
            c += 1
    else:
        # save only the finished map
        plt.scatter(x_list[::jump_steps], y_list[::jump_steps], s=size_scatter, c=colormap)
        plt.axis('off')
        name = "./map/" + "map_all" + ".png"
        plt.savefig(name)

    #plt.imshow(data)
    #plt.axis('off')
    #plt.show()
    
def create_track_arrow(txt_path):
    """ Create arrow diagramms to show the direction heading. """
    x_list, y_list = read_txt_pos_track(txt_path)
    
    prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
            shrinkA=0,shrinkB=0)
    
    plt_sz = jump_steps * 25
    old_x, old_y = x_list[0], y_list[0]
    c = 0
    for x, y in zip(x_list[1::jump_steps], y_list[1::jump_steps]):
        plt.figure()
        plt.axis((-plt_sz, plt_sz, -plt_sz, plt_sz))
        plt.gca().invert_yaxis()
        x_dif = x - old_x
        y_dif = y - old_y
        old_x, old_y = x, y
        plt.annotate("", xy=(x_dif,y_dif), xytext=(0,0), arrowprops=prop)
        name = "./arrows/" + "arrow" + str(c) + ".png"
        print("x_dif:", x_dif, "y_dif", y_dif, "counter", c)
        c += 1
        plt.savefig(name)
        
    # plt.axis('off')
    # plt.show()
    
def create_mosaic(map_path, arrow_path, view_path, length=0):
    """ Create mosaic of the map, arrow and view of the agent. """
    
    for i in range(0, length):
        print("counter", i)
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic([['map', 'arrow'],['map', 'view']], gridspec_kw={'width_ratios':[2, 1]})
        axs['map'].set_title('Map')
        axs['arrow'].set_title('Direction')
        axs['view'].set_title('Agents view')
    
        axs['map'].imshow(image.imread(map_path + "map" + str(i) + ".png"))
        axs['arrow'].imshow(image.imread(arrow_path + "arrow" + str(i) + ".png"))
        axs['view'].imshow(image.imread(view_path + str(i*jump_steps) + ".png")) ### fixed
        
        text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1')
                
        axs['map'].axis('off')
        axs['arrow'].axis('off')
        axs['view'].axis('off')

        name = mosaic_path + "mosaic" + str(i) + ".png"
        plt.savefig(name)

if __name__ == "__main__":
    args = arguments_given()
    
    path_txt_pos_track = args.use_tta
    image_path = args.image_path
    map_path = args.map_path
    arrow_path = args.arrow_path
    view_path = args.view_path
    mosaic_path = args.mosaic_path
    frame_no = args.frame_no
    jump_steps = args.jump_steps
    map_x_minus = args.map_x_minus
    map_y_plus = args.map_y_plus
    
    
    create_track_on_map(path_txt_pos_track, image_path, True)
    create_track_arrow(path_txt_pos_track)
        
    create_mosaic(map_path, arrow_path, view_path, frame_no)







