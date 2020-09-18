#exec(open("changememe.py").read()) 
"""
Coded using python 3.8.3 in the anaconda3 distribution 
The following packages were used:
numpy 1.18.2
matplotlib 3.2.2
PIL 6.2.2
scipy 1.5.0
tkinter
copy
random
cx_Freeze 6.2 (for compiling)

changes to make:

-write readme
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import random
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter.filedialog import asksaveasfile
import copy
import os
import glob


def apply_noise(data,noise_mag):
    noise = np.random.rand(np.shape(data)[0],np.shape(data)[1],np.shape(data)[2])*255*noise_mag
    data = data+noise
    data = normalize_data(data)
    return data

def apply_lpf(data,lpf_max):

    data = normalize_data(data)
    
    f_cut = int( np.min(np.shape(data)[0:2]) * lpf_max/2 )

    data_fft0 = np.fft.fft2(data[:,:,0])
    data_fft0 = np.fft.fftshift(data_fft0)

    data_fft0[:,0:f_cut] = 0
    data_fft0[:,-f_cut:np.shape(data)[1]] = 0
    data_fft0[0:f_cut,:] = 0
    data_fft0[-f_cut:np.shape(data)[0],:] = 0
    
    data_fft0 = np.fft.ifftshift(data_fft0)
    
    data_fft1 = np.fft.fft2(data[:,:,1])
    data_fft1 = np.fft.fftshift(data_fft1)

    data_fft1[:,0:f_cut] = 0
    data_fft1[:,-f_cut:np.shape(data)[1]] = 0
    data_fft1[0:f_cut,:] = 0
    data_fft1[-f_cut:np.shape(data)[0],:] = 0
    
    data_fft1 = np.fft.ifftshift(data_fft1)
    
    data_fft2 = np.fft.fft2(data[:,:,2])
    data_fft2 = np.fft.fftshift(data_fft2)

    data_fft2[:,0:f_cut] = 0
    data_fft2[:,-f_cut:np.shape(data)[1]] = 0
    data_fft2[0:f_cut,:] = 0
    data_fft2[-f_cut:np.shape(data)[0],:] = 0
    
    data_fft2 = np.fft.ifftshift(data_fft2)
    
    data0 = np.real(np.fft.ifft2(data_fft0))
    data1 = np.real(np.fft.ifft2(data_fft1))
    data2 = np.real(np.fft.ifft2(data_fft2))
    
    data = np.stack([data0,data1,data2],axis=-1)
    data = normalize_data(data)
    return data
    
def apply_vert_bar(data,vert_bar_shift_mag,vert_bar_width):
    for k in range(int(np.shape(data)[1]/vert_bar_width)):
        shift_k = int(random.random()*vert_bar_shift_mag - vert_bar_shift_mag/2)
        data[:,k*vert_bar_width:(k+1)*vert_bar_width,:] = np.roll(data[:,k*vert_bar_width:(k+1)*vert_bar_width,:],shift_k,axis=0)

    return data
    
def apply_horz_bar(data,horz_bar_shift_mag,horz_bar_width):
    for k in range(int(np.shape(data)[0]/horz_bar_width)):
        shift_k = int(random.random()*horz_bar_shift_mag - horz_bar_shift_mag/2)
        data[k*horz_bar_width:(k+1)*horz_bar_width,:,:] = np.roll(data[k*horz_bar_width:(k+1)*horz_bar_width,:,:],shift_k,axis=1)

    return data
    
def apply_col_scramble(data,col_scramble_bin_width,frac_col_to_scramble):
    
    num_to_scramble = int(round(frac_col_to_scramble*col_scramble_bin_width))
    
    if num_to_scramble>0:
        for k in range(int(np.shape(data)[1]/col_scramble_bin_width)):
            inds = []
            for j in range(num_to_scramble):
                ind = int(random.random()*col_scramble_bin_width)
                if (k*col_scramble_bin_width+ind)>np.shape(data)[1]:
                    ind = np.shape(data)[1] - 1
                inds = inds + [ind]
            temp = data[:,k*col_scramble_bin_width+inds[0],:]
            for j in range(num_to_scramble-1):
                data[:,k*col_scramble_bin_width+inds[j],:] = data[:,k*col_scramble_bin_width+inds[j+1],:]
            data[:,k*col_scramble_bin_width+inds[len(inds)-1],:] = temp
    return data
   
def apply_row_scramble(data,row_scramble_bin_width,frac_row_to_scramble):
    
    num_to_scramble = int(round(frac_row_to_scramble*row_scramble_bin_width))
    
    if num_to_scramble>0:
        for k in range(int(np.shape(data)[0]/row_scramble_bin_width)):
            inds = []
            for j in range(num_to_scramble):
                ind = int(random.random()*row_scramble_bin_width)
                if (k*row_scramble_bin_width+ind)>np.shape(data)[0]:
                    ind = np.shape(data)[0] - 1
                inds = inds + [ind]
            temp = data[k*row_scramble_bin_width+inds[0],:,:]
            for j in range(num_to_scramble-1):
                data[k*row_scramble_bin_width+inds[j],:,:] = data[k*row_scramble_bin_width+inds[j+1],:,:]
            data[k*row_scramble_bin_width+inds[len(inds)-1],:,:] = temp
    return data

def apply_rand_black_squares(data,black_square_coverage,num_black_squares):
    square_width = int(np.sqrt(np.shape(data)[0]*np.shape(data)[1]*black_square_coverage/num_black_squares))
    for k in range(num_black_squares):
        y = int(random.random()*np.shape(data)[0])
        x = int(random.random()*np.shape(data)[1])
        data[y-int(square_width/2):y+int(square_width/2),x-int(square_width/2):x+int(square_width/2),:]=0
        if np.shape(data)[2]==4:
            data[y-int(square_width/2):y+int(square_width/2),x-int(square_width/2):x+int(square_width/2),3]=255
    return data
        
def apply_Q_squares(data,version=2,black_square_coverage = 0.2):
    square_width = int(np.sqrt(np.shape(data)[0]*np.shape(data)[1]*black_square_coverage/4))
    
    if version==1:
        data[0:square_width,0:square_width,:]=0
        data[-square_width:np.shape(data)[0],-square_width:np.shape(data)[1],:]=0
        data[0:square_width,-square_width:np.shape(data)[1],:]=0
        data[-square_width:np.shape(data)[0],0:square_width,:]=0
        if np.shape(data)[2]==4:
            data[0:square_width,0:square_width,3]=255
            data[-square_width:np.shape(data)[0],-square_width:np.shape(data)[1],3]=255
            data[0:square_width,-square_width:np.shape(data)[1],3]=255
            data[-square_width:np.shape(data)[0],0:square_width,3]=255
    elif version==2:
        y_mid = int(np.shape(data)[0]/2)
        x_mid = int(np.shape(data)[1]/2)
        hw=int(square_width/2)
        data[y_mid-hw:y_mid+hw,0:square_width,:]=0
        data[0:square_width,x_mid-hw:x_mid+hw,:]=0
        data[y_mid-hw:y_mid+hw,np.shape(data)[1]-square_width:np.shape(data)[1],:]=0
        data[np.shape(data)[0]-square_width:np.shape(data)[0],x_mid-hw:x_mid+hw,:]=0
        if np.shape(data)[2]==4:
            data[y_mid-hw:y_mid+hw,0:square_width,3]=255
            data[0:square_width,x_mid-hw:x_mid+hw,3]=255
            data[y_mid-hw:y_mid+hw,np.shape(data)[1]-square_width:np.shape(data)[1],3]=255
            data[np.shape(data)[0]-square_width:np.shape(data)[0],x_mid-hw:x_mid+hw,3]=255
    
    return data
        
def apply_ghost_image(data,ghost_image_filename,ghost_image_mag):
    ghost_image = Image.open("temp_images/ghost_img."+ghost_image_filename.split(".")[1])
    ghost_data = np.array(ghost_image)
    
    data = normalize_data(data)
    ghost_data = normalize_data(ghost_data)
    
    mindim = [1,1]
    mindim[0] = np.min([np.shape(data)[0],np.shape(ghost_data)[0]])
    mindim[1] = np.min([np.shape(data)[1],np.shape(ghost_data)[1]])
    
    startind = [1,1]
    startind[0] = int((np.shape(data)[0]-mindim[0])/2)
    startind[1] = int((np.shape(data)[1]-mindim[1])/2)
    
    startindg = [1,1]
    startindg[0] = int((np.shape(ghost_data)[0]-mindim[0])/2)
    startindg[1] = int((np.shape(ghost_data)[1]-mindim[1])/2)
    
    data[startind[0]:startind[0]+mindim[0],startind[1]:startind[1]+mindim[1],0:3] = data[startind[0]:startind[0]+mindim[0],startind[1]:startind[1]+mindim[1],0:3] + ghost_image_mag*ghost_data[startindg[0]:startindg[0]+mindim[0],startindg[1]:startindg[1]+mindim[1],0:3]
    data = normalize_data(data)
    return data
    
def apply_random_transformation(data):
    
    num_transformations = 8
    transformation_probability = 0.5
    
    #magnitudes and max values
    noise_mag_min = 0.0 #noise magnitude as fraction of max intensity (0 to 1)
    noise_mag_max = 0.5
    lpf_min = 0.0
    lpf_max = 0.7 #intensity of lpf, value from 0 to 1
    vert_bar_shift_min = 0 #max shift for vertical bars
    vert_bar_shift_max = 50
    vert_bar_width_min = 1 #vertical bar width for vertical bar shift
    vert_bar_width_max = 200
    horz_bar_shift_min = 0 
    horz_bar_shift_max = 50
    horz_bar_width_min = 1 
    horz_bar_width_max = 200
    col_scramble_bin_width_min = 2
    col_scramble_bin_width_max = 20
    frac_col_to_scramble_min = 0.0
    frac_col_to_scramble_max = 1.0
    row_scramble_bin_width_min = 2
    row_scramble_bin_width_max = 20
    frac_row_to_scramble_min = 0.0
    frac_row_to_scramble_max = 1.0
    black_square_coverage_min = 0.0
    black_square_coverage_max = 0.2
    num_black_squares_min = 1
    num_black_squares_max = 200
    Q_square_coverage_min = 0.0
    Q_square_coverage_max = 0.2
    
    noise_mag = noise_mag_min + random.random()*(noise_mag_max-noise_mag_min)
    lpf_cutoff = lpf_min + random.random()*(lpf_max-lpf_min)
    vert_bar_shift = int(round(vert_bar_shift_min + random.random()*(vert_bar_shift_max-vert_bar_shift_min)))
    vert_bar_width = int(round(vert_bar_width_min + random.random()*(vert_bar_width_max-vert_bar_width_min)))
    horz_bar_shift = int(round(horz_bar_shift_min + random.random()*(horz_bar_shift_max-horz_bar_shift_min)))
    horz_bar_width = int(round(horz_bar_width_min + random.random()*(horz_bar_width_max-horz_bar_width_min)))
    col_scramble_bin_width = int(round(col_scramble_bin_width_min + random.random()*(col_scramble_bin_width_max-col_scramble_bin_width_min)))
    frac_col_to_scramble = frac_col_to_scramble_min + random.random()*(frac_col_to_scramble_max - frac_col_to_scramble_min)
    row_scramble_bin_width = int(round(row_scramble_bin_width_min + random.random()*(row_scramble_bin_width_max-row_scramble_bin_width_min)))
    frac_row_to_scramble = frac_row_to_scramble_min + random.random()*(frac_row_to_scramble_max - frac_row_to_scramble_min)
    black_square_coverage = black_square_coverage_min + random.random()*(black_square_coverage_max-black_square_coverage_min)
    num_black_squares = int(round(num_black_squares_min + random.random()*(num_black_squares_max-num_black_squares_min)))
    Q_square_version = int(round(random.random()*1.0)+1)
    Q_square_coverage = Q_square_coverage_min + random.random()*(Q_square_coverage_max - Q_square_coverage_min)
    
    inds = list(range(num_transformations))
    random.shuffle(inds)
    
    for k in inds:
        if (k==0) and (random.random() < transformation_probability):
            data = apply_noise(data,noise_mag)
        if (k==1) and (random.random() < transformation_probability):
            data = apply_lpf(data,lpf_cutoff)
        if (k==2) and (random.random() < transformation_probability):
            data = apply_vert_bar(data,vert_bar_shift,vert_bar_width)
        if (k==3) and (random.random() < transformation_probability):
            data = apply_col_scramble(data,col_scramble_bin_width,frac_col_to_scramble)
        if (k==4) and (random.random() < transformation_probability):
            data = apply_rand_black_squares(data,black_square_coverage,num_black_squares)
        if (k==5) and (random.random() < transformation_probability/2):
            data = apply_Q_squares(data,Q_square_version,Q_square_coverage)
        if (k==6) and (random.random() < transformation_probability):
            data = apply_horz_bar(data,horz_bar_shift,horz_bar_width)
        if (k==7) and (random.random() < transformation_probability):
            data = apply_row_scramble(data,row_scramble_bin_width,frac_row_to_scramble)
    
    return data
    
def update_preview(data,inc_flag=True):
    global img_data
    global img_loaded_flag
    global img_version
    global img_address
    if inc_flag:
        img_version += 1
    img_data = data
    img_loaded_flag = True
    image = Image.fromarray(data.astype(np.uint8))
    mpl.rcParams['toolbar'] = 'None' 
    fig=plt.figure(1)
    fig.clear()
    plt.imshow(image)
    plt.title("preview")
    plt.draw()
    if inc_flag:
        save_file(data,"temp_images/img_"+str(img_version)+"."+img_address.split(".")[1])

def normalize_data(data):
    data = data*(255/np.max(data))
    data = data.astype(int)
    return data

def load_file(filename):
    image = Image.open(filename)
    data = np.array(image)
    return data
    
def save_file(data,filename):
    image = Image.fromarray(data.astype(np.uint8))
    image.save(filename)
    
def set_sliders_to_defaults():
    noise_mag = 0.3
    lpf_cutoff = 0.7
    vert_bar_shift = 40
    vert_bar_width = 20
    horz_bar_shift = 40
    horz_bar_width = 20
    col_scramble_bin_width = 8
    frac_col_to_scramble = 1.0
    row_scramble_bin_width = 8
    frac_row_to_scramble = 1.0
    black_square_coverage = 0.1
    num_black_squares = 50
    ghost_image_mag = 0.3
    Q_square_version = 1
    Q_square_coverage = 0.2
    
    noise_mag_slider.set(noise_mag)
    lowpass_cut_slider.set(lpf_cutoff)
    vert_bar_shift_slider.set(vert_bar_shift)
    vert_bar_width_slider.set(vert_bar_width)
    horz_bar_shift_slider.set(vert_bar_shift)
    horz_bar_width_slider.set(vert_bar_width)
    col_scramble_bin_width_slider.set(col_scramble_bin_width)
    frac_col_slider.set(frac_col_to_scramble)
    row_scramble_bin_width_slider.set(row_scramble_bin_width)
    frac_row_slider.set(frac_row_to_scramble)
    rand_black_square_coverage_slider.set(black_square_coverage)
    rand_black_square_num_slider.set(num_black_squares)
    ghost_image_mag_slider.set(ghost_image_mag)
    Q_squares_version_slider.set(Q_square_version)
    Q_squares_coverage_slider.set(Q_square_coverage)
    
def save_as(): 
    files = [("PNG files","*.png"),  
             ("jpg files","*.jpg"), 
             ('All Files', '*.*')] 
    fileinfo = asksaveasfile(filetypes = files, defaultextension = files)
    
    filename_flag = True
    try:
        fileinfo.name
    except:
        filename_flag = False
        
    if filename_flag:
        filename=fileinfo.name
        global img_data
        image = Image.fromarray(img_data.astype(np.uint8))
        image.save(filename)
    
def browseFiles(): 
    global img_address
    global img_version
    
    filename = tk.filedialog.askopenfilename(initialdir = "/", 
        title = "Select a File", 
        filetypes = (("all files", "*.*"),("jpg files","*.jpg"),("PNG files","*.png")))
    
    if filename != '':
        img_address = copy.deepcopy(filename)
        data=load_file(filename)
        # Change label contents 
        label_file_explorer.configure(text="File Opened: \n"+filename+"\nimage size: "+str(np.shape(data))) 
        
        update_preview(data,inc_flag=False)
        img_version=0
        clear_undo_imgs()
        save_file(data,"temp_images/orig_img."+filename.split(".")[1])
        
    else:
        print("error: no file found")
        
def browseFiles_ghost(): 
    global img_address_ghost
    global ghost_img_loaded_flag
    
    filename = tk.filedialog.askopenfilename(initialdir = "/", 
        title = "Select a File", 
        filetypes = (("all files", "*.*"),("jpg files","*.jpg"),("PNG files","*.png")))
    
    if filename != '':
        img_address_ghost = filename
        data=load_file(filename)
        # Change label contents 
        label_ghost_file_explorer.configure(text="File Opened: \n"+filename+"\nimage size: "+str(np.shape(data))) 
        
        save_file(data,"temp_images/ghost_img."+filename.split(".")[1])
        ghost_img_loaded_flag = True
    else:
        print("error: no file found")
    
def reset_button_handle(event):
    global img_address
    global img_data
    global img_loaded_flag
    global img_version
    if img_loaded_flag:
        img_data=load_file("temp_images/orig_img."+img_address.split(".")[1])
        img_version = 0
        update_preview(img_data,inc_flag=False) 
        clear_undo_imgs()
        
def undo_button_handle(event):
    global img_data
    global img_address
    global img_loaded_flag
    global img_version
    if img_loaded_flag:
        img_version -= 1
        if img_version<0:
            img_version=0
        if img_version==0:
            img_data = load_file("temp_images/orig_img."+img_address.split(".")[1])
        else:
            img_data = load_file("temp_images/img_"+str(img_version)+"."+img_address.split(".")[1])
        update_preview(img_data,inc_flag=False)
        
def redo_button_handle(event):
    global img_data
    global img_address
    global img_loaded_flag
    global img_version
    if img_loaded_flag:
        if os.path.exists("temp_images/img_"+str(img_version+1)+"."+img_address.split(".")[1]):
            img_version += 1
            img_data = load_file("temp_images/img_"+str(img_version)+"."+img_address.split(".")[1])
            update_preview(img_data,inc_flag=False)
        
        
def reset_sliders_button_handle(event):
    set_sliders_to_defaults()
    
def apply_noise_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_noise(img_data,noise_mag_slider.get())
        update_preview(data)
    
def apply_lpf_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_lpf(img_data,lowpass_cut_slider.get())
        update_preview(data)
        
def apply_vert_bar_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_vert_bar(img_data,vert_bar_shift_slider.get(),vert_bar_width_slider.get())
        update_preview(data)

def apply_horz_bar_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_horz_bar(img_data,horz_bar_shift_slider.get(),horz_bar_width_slider.get())
        update_preview(data)
        
def apply_col_scramble_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_col_scramble(img_data,col_scramble_bin_width_slider.get(),frac_col_slider.get())
        update_preview(data)

def apply_row_scramble_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_row_scramble(img_data,row_scramble_bin_width_slider.get(),frac_row_slider.get())
        update_preview(data)
        
def apply_rand_black_square_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_rand_black_squares(img_data,rand_black_square_coverage_slider.get(),rand_black_square_num_slider.get())
        update_preview(data)
        
def apply_Q_squares_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_Q_squares(img_data,Q_squares_version_slider.get(),Q_squares_coverage_slider.get())
        update_preview(data)
        
def apply_ghost_image_button_handle(event):
    global img_data
    global img_address_ghost
    global img_loaded_flag
    global ghost_img_loaded_flag
    if img_loaded_flag and ghost_img_loaded_flag:
        data = apply_ghost_image(img_data,img_address_ghost,ghost_image_mag_slider.get())
        update_preview(data)

def apply_random_transformation_button_handle(event):
    global img_data
    global img_loaded_flag
    if img_loaded_flag:
        data = apply_random_transformation(img_data)
        update_preview(data)

def row_num(position_num,num_cols):
    return int(np.floor(position_num/num_cols))
    
def col_num(position_num,num_cols):
    return int(position_num%num_cols)

def clear_temp_imgs():
    files = glob.glob('temp_images/*')
    for f in files:
        os.remove(f)
        
def clear_undo_imgs():
    files = glob.glob('temp_images/*')
    for f in files:
        if "img_" in f:
            os.remove(f)
        
if __name__ == '__main__':
    
    global img_loaded_flag
    global ghost_img_loaded_flag
    global img_data
    global img_version
    img_data = None
    img_loaded_flag = False
    ghost_img_loaded_flag = False
    img_version = 0
    
    mpl.use('TkAgg')
    
    #if not os.path.exists('temp_images'):
    #    os.makedirs('temp_images')
    clear_temp_imgs()
    
    #magnitudes and max values
    noise_mag_min = 0.0 #noise magnitude as fraction of max intensity (0 to 1)
    noise_mag_max = 3.0
    lpf_min = 0.0
    lpf_max = 1.0 #intensity of lpf, value from 0 to 1
    vert_bar_shift_min = 0 #max shift for vertical bars
    vert_bar_shift_max = 200
    vert_bar_width_min = 1 #vertical bar width for vertical bar shift
    vert_bar_width_max = 200
    horz_bar_shift_min = 0 
    horz_bar_shift_max = 200
    horz_bar_width_min = 1 
    horz_bar_width_max = 200
    col_scramble_bin_width_min = 2
    col_scramble_bin_width_max = 100
    frac_col_to_scramble_min = 0.0
    frac_col_to_scramble_max = 1.0
    row_scramble_bin_width_min = 2
    row_scramble_bin_width_max = 100
    frac_row_to_scramble_min = 0.0
    frac_row_to_scramble_max = 1.0
    black_square_coverage_min = 0.0
    black_square_coverage_max = 0.5
    num_black_squares_min = 1
    num_black_squares_max = 200
    ghost_image_filename = "../groyper cig.jpg"
    ghost_image_mag_min = 0.0
    ghost_image_mag_max = 1.0
    
    header_font = 'Helvetica 12 bold'
    main_menu_font = 'Helvetica 15'
    slider_length = 150
    toggle_length = 50
    
    transform_frame_num_rows = 1
    transform_frame_num_cols = 9
    
    noise_position = 2
    lpf_position = 3
    vert_bar_position = 5
    horz_bar_position = 4
    col_scramble_position = 7
    row_scramble_position = 6
    rand_black_square_position = 1
    Q_squares_position = 0
    ghost_image_position = 8
    
    window = tk.Tk()
    window.title("MemeScramble")
    
    main_frame = tk.Frame(master=window, width=400, height=500)
    main_frame.pack_propagate(0)
    main_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    main_menu_frame = tk.Frame(master=main_frame, width=400, height=400)
    main_menu_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    
    # Special type of "canvas" to allow for matplotlib graphing
    fig = plt.figure(1)
    canvas = mpl.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=main_frame)
    plot_widget = canvas.get_tk_widget()
    plot_widget.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    transform_frame = tk.Frame(master=window, width=600)
    for i in range(transform_frame_num_cols):
        transform_frame.columnconfigure(i, weight=1, minsize=200)
    for i in range(transform_frame_num_rows):
        transform_frame.rowconfigure(i, weight=1, minsize=250)
    transform_frame.pack(fill=tk.BOTH)

    noise_frame = tk.Frame(master=transform_frame)
    noise_frame.grid(row=row_num(noise_position,transform_frame_num_cols),column=col_num(noise_position,transform_frame_num_cols),sticky='n')
    noise_label = tk.Label(master=noise_frame, text="Random Noise", font=header_font)
    noise_label.pack()
    noise_mag_slider = tk.Scale(master=noise_frame,from_=noise_mag_min,to=noise_mag_max,orient="horizontal",resolution=0.1,length=slider_length)
    noise_mag_slider.pack(pady=5)
    noise_mag_slider_label = tk.Label(master=noise_frame, text="noise magnitude\n(as % of image mag)")
    noise_mag_slider_label.pack()
    apply_noise_button = tk.Button(master=noise_frame,text="Apply Random Noise")
    apply_noise_button.bind("<Button-1>",apply_noise_button_handle)
    apply_noise_button.pack(pady=15)


    lowpass_filter_frame = tk.Frame(master=transform_frame)
    lowpass_filter_frame.grid(row=row_num(lpf_position,transform_frame_num_cols),column=col_num(lpf_position,transform_frame_num_cols),sticky='n')
    lowpass_filter_label = tk.Label(master=lowpass_filter_frame, text="Low Pass Filter", font=header_font)
    lowpass_filter_label.pack()
    lowpass_cut_slider = tk.Scale(master=lowpass_filter_frame,from_=lpf_min,to=lpf_max,orient="horizontal",resolution=0.1,length=slider_length)
    lowpass_cut_slider.pack(pady=5)
    lowpass_cut_slider_label = tk.Label(master=lowpass_filter_frame, text="% frequencies to cut")
    lowpass_cut_slider_label.pack()
    apply_lpf_button = tk.Button(master=lowpass_filter_frame,text="Apply Low Pass Filter")
    apply_lpf_button.bind("<Button-1>",apply_lpf_button_handle)
    apply_lpf_button.pack(pady=15)
    
    vert_bar_frame = tk.Frame(master=transform_frame)
    vert_bar_frame.grid(row=row_num(vert_bar_position,transform_frame_num_cols),column=col_num(vert_bar_position,transform_frame_num_cols),sticky='n')
    vert_bar_label = tk.Label(master=vert_bar_frame, text="Column Shift", font=header_font)
    vert_bar_label.pack()
    vert_bar_shift_slider = tk.Scale(master=vert_bar_frame,from_=vert_bar_shift_min,to=vert_bar_shift_max,orient="horizontal",resolution=1,length=slider_length)
    vert_bar_shift_slider.pack(pady=5)
    vert_bar_shift_slider_label = tk.Label(master=vert_bar_frame, text="Shift Magnitude")
    vert_bar_shift_slider_label.pack()
    vert_bar_width_slider = tk.Scale(master=vert_bar_frame,from_=vert_bar_width_min,to=vert_bar_width_max,orient="horizontal",resolution=1,length=slider_length)
    vert_bar_width_slider.pack(pady=5)
    vert_bar_width_slider_label = tk.Label(master=vert_bar_frame, text="Column Width")
    vert_bar_width_slider_label.pack()
    apply_vert_bar_button = tk.Button(master=vert_bar_frame,text="Apply Column Shift")
    apply_vert_bar_button.bind("<Button-1>",apply_vert_bar_button_handle)
    apply_vert_bar_button.pack(pady=15)
    
    horz_bar_frame = tk.Frame(master=transform_frame)
    horz_bar_frame.grid(row=row_num(horz_bar_position,transform_frame_num_cols),column=col_num(horz_bar_position,transform_frame_num_cols),sticky='n')
    horz_bar_label = tk.Label(master=horz_bar_frame, text="Row Shift", font=header_font)
    horz_bar_label.pack()
    horz_bar_shift_slider = tk.Scale(master=horz_bar_frame,from_=horz_bar_shift_min,to=horz_bar_shift_max,orient="horizontal",resolution=1,length=slider_length)
    horz_bar_shift_slider.pack(pady=5)
    horz_bar_shift_slider_label = tk.Label(master=horz_bar_frame, text="Shift Magnitude")
    horz_bar_shift_slider_label.pack()
    horz_bar_width_slider = tk.Scale(master=horz_bar_frame,from_=horz_bar_width_min,to=horz_bar_width_max,orient="horizontal",resolution=1,length=slider_length)
    horz_bar_width_slider.pack(pady=5)
    horz_bar_width_slider_label = tk.Label(master=horz_bar_frame, text="Row Width")
    horz_bar_width_slider_label.pack()
    apply_horz_bar_button = tk.Button(master=horz_bar_frame,text="Apply Row Shift")
    apply_horz_bar_button.bind("<Button-1>",apply_horz_bar_button_handle)
    apply_horz_bar_button.pack(pady=15)
    
    col_scramble_frame = tk.Frame(master=transform_frame)
    col_scramble_frame.grid(row=row_num(col_scramble_position,transform_frame_num_cols),column=col_num(col_scramble_position,transform_frame_num_cols),sticky='n')
    col_scramble_label = tk.Label(master=col_scramble_frame, text="Column Scramble", font=header_font)
    col_scramble_label.pack()
    col_scramble_bin_width_slider = tk.Scale(master=col_scramble_frame,from_=col_scramble_bin_width_min,to=col_scramble_bin_width_max,orient="horizontal",resolution=1,length=slider_length)
    col_scramble_bin_width_slider.pack(pady=5)
    col_scramble_bin_width_slider_label = tk.Label(master=col_scramble_frame, text="Bin Width")
    col_scramble_bin_width_slider_label.pack()
    frac_col_slider = tk.Scale(master=col_scramble_frame,from_=frac_col_to_scramble_min,to=frac_col_to_scramble_max,orient="horizontal",resolution=0.25,length=slider_length)
    frac_col_slider.pack(pady=5)
    frac_col_slider_label = tk.Label(master=col_scramble_frame, text="% to scramble")
    frac_col_slider_label.pack()
    apply_col_scramble_button = tk.Button(master=col_scramble_frame,text="Apply Column Scramble")
    apply_col_scramble_button.bind("<Button-1>",apply_col_scramble_button_handle)
    apply_col_scramble_button.pack(pady=15)
    
    row_scramble_frame = tk.Frame(master=transform_frame)
    row_scramble_frame.grid(row=row_num(row_scramble_position,transform_frame_num_cols),column=col_num(row_scramble_position,transform_frame_num_cols),sticky='n')
    row_scramble_label = tk.Label(master=row_scramble_frame, text="Row Scramble", font=header_font)
    row_scramble_label.pack()
    row_scramble_bin_width_slider = tk.Scale(master=row_scramble_frame,from_=row_scramble_bin_width_min,to=row_scramble_bin_width_max,orient="horizontal",resolution=1,length=slider_length)
    row_scramble_bin_width_slider.pack(pady=5)
    row_scramble_bin_width_slider_label = tk.Label(master=row_scramble_frame, text="Bin Width")
    row_scramble_bin_width_slider_label.pack()
    frac_row_slider = tk.Scale(master=row_scramble_frame,from_=frac_col_to_scramble_min,to=frac_col_to_scramble_max,orient="horizontal",resolution=0.25,length=slider_length)
    frac_row_slider.pack(pady=5)
    frac_row_slider_label = tk.Label(master=row_scramble_frame, text="% to scramble")
    frac_row_slider_label.pack()
    apply_row_scramble_button = tk.Button(master=row_scramble_frame,text="Apply Row Scramble")
    apply_row_scramble_button.bind("<Button-1>",apply_row_scramble_button_handle)
    apply_row_scramble_button.pack(pady=15)
    
    rand_black_square_frame = tk.Frame(master=transform_frame)
    rand_black_square_frame.grid(row=row_num(rand_black_square_position,transform_frame_num_cols),column=col_num(rand_black_square_position,transform_frame_num_cols),sticky='n')
    rand_black_square_label = tk.Label(master=rand_black_square_frame, text="Random Black Squares", font=header_font)
    rand_black_square_label.pack()
    rand_black_square_coverage_slider = tk.Scale(master=rand_black_square_frame,from_=black_square_coverage_min,to=black_square_coverage_max,orient="horizontal",resolution=0.01,length=slider_length)
    rand_black_square_coverage_slider.pack(pady=5)
    rand_black_square_coverage_slider_label = tk.Label(master=rand_black_square_frame, text="Black Square Coverage")
    rand_black_square_coverage_slider_label.pack()
    rand_black_square_num_slider = tk.Scale(master=rand_black_square_frame,from_=num_black_squares_min,to=num_black_squares_max,orient="horizontal",resolution=1,length=slider_length)
    rand_black_square_num_slider.pack(pady=5)
    rand_black_square_num_slider_label = tk.Label(master=rand_black_square_frame, text="Number of Black Squares")
    rand_black_square_num_slider_label.pack()
    apply_rand_black_square_button = tk.Button(master=rand_black_square_frame,text="Apply Random Black Squares")
    apply_rand_black_square_button.bind("<Button-1>",apply_rand_black_square_button_handle)
    apply_rand_black_square_button.pack(pady=15)
    
    Q_squares_frame = tk.Frame(master=transform_frame)
    Q_squares_frame.grid(row=row_num(Q_squares_position,transform_frame_num_cols),column=col_num(Q_squares_position,transform_frame_num_cols),sticky='n')
    Q_squares_label = tk.Label(master=Q_squares_frame, text="Q Squares", font=header_font)
    Q_squares_label.pack()
    Q_squares_version_slider = tk.Scale(master=Q_squares_frame,from_=1,to=2,orient="horizontal",resolution=1,length=toggle_length)
    Q_squares_version_slider.pack(pady=5)
    Q_squares_version_slider_label = tk.Label(master=Q_squares_frame, text="Q square version")
    Q_squares_version_slider_label.pack()
    Q_squares_coverage_slider = tk.Scale(master=Q_squares_frame,from_=black_square_coverage_min,to=black_square_coverage_max,orient="horizontal",resolution=0.01,length=slider_length)
    Q_squares_coverage_slider.pack(pady=5)
    Q_squares_coverage_slider_label = tk.Label(master=Q_squares_frame, text="Square Coverage")
    Q_squares_coverage_slider_label.pack()
    apply_Q_squares_button = tk.Button(master=Q_squares_frame,text="Apply Q Squares")
    apply_Q_squares_button.bind("<Button-1>",apply_Q_squares_button_handle)
    apply_Q_squares_button.pack(pady=15)
    
    ghost_image_frame = tk.Frame(master=transform_frame)
    ghost_image_frame.grid(row=row_num(ghost_image_position,transform_frame_num_cols),column=col_num(ghost_image_position,transform_frame_num_cols),sticky='n')
    ghost_image_label = tk.Label(master=ghost_image_frame, text="Ghost Image\n", font=header_font)
    ghost_image_label.pack()
    label_ghost_file_explorer = tk.Label(master=ghost_image_frame,  
                                text = "load a file for ghost image") 
       
    button_explore_ghost = tk.Button(master=ghost_image_frame,  
                            text = "Browse Files", 
                            command = browseFiles_ghost)
    label_ghost_file_explorer.pack()
    button_explore_ghost.pack()
    ghost_image_mag_slider = tk.Scale(master=ghost_image_frame,from_=ghost_image_mag_min,to=ghost_image_mag_max,orient="horizontal",resolution=0.01,length=slider_length)
    ghost_image_mag_slider.pack(pady=5)
    ghost_image_mag_slider_label = tk.Label(master=ghost_image_frame, text="ghost image magnitude\n(as % of image mag)")
    ghost_image_mag_slider_label.pack()
    apply_ghost_image_button = tk.Button(master=ghost_image_frame,text="Apply Ghost Image")
    apply_ghost_image_button.bind("<Button-1>",apply_ghost_image_button_handle)
    apply_ghost_image_button.pack(pady=15)


    greeting = tk.Label(master=main_menu_frame,text="MemeScramble\n", font=header_font)
    greeting.pack()

    label_file_explorer = tk.Label(master=main_menu_frame,  
                                text = "load a file") 
       
    button_explore = tk.Button(master=main_menu_frame,  
                            text = "Browse Files", 
                            command = browseFiles, 
                            font=main_menu_font)  
       
    label_file_explorer.pack()
    button_explore.pack(pady=5)

    reset_button = tk.Button(master=main_menu_frame,text="Start Over", font=main_menu_font,fg="red")
    reset_button.bind("<Button-1>",reset_button_handle)
    reset_button.pack(pady=5)
    
    undo_button = tk.Button(master=main_menu_frame,text="Undo", font=main_menu_font,fg="red")
    undo_button.bind("<Button-1>",undo_button_handle)
    undo_button.pack(pady=5)
    
    redo_button = tk.Button(master=main_menu_frame,text="Redo", font=main_menu_font)
    redo_button.bind("<Button-1>",redo_button_handle)
    redo_button.pack(pady=5)
    
    save_button = tk.Button(master=main_menu_frame, text = 'Save Result As', command = lambda : save_as(), font=main_menu_font, fg="green")
    save_button.pack(pady=5)
    
    reset_sliders_button = tk.Button(master=main_menu_frame,text="Reset Settings to Defaults")
    reset_sliders_button.bind("<Button-1>",reset_sliders_button_handle)
    reset_sliders_button.pack(pady=5)
    
    apply_random_transformation_button = tk.Button(master=main_menu_frame,text="Apply Random Transformation")
    apply_random_transformation_button.bind("<Button-1>",apply_random_transformation_button_handle)
    apply_random_transformation_button.pack(pady=5)
    
    exit_button = tk.Button(master=main_menu_frame,text="Exit", font=main_menu_font,fg="red",command=window.quit).pack(pady=5)
    
    set_sliders_to_defaults()
    
    window.mainloop()

















































