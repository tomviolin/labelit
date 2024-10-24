#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Major decision:
    default internal representation of all coordinates shall be
    in normalized units of the frame size.  This will allow
    the program to work with any frame size, and will allow
    the program to work with any display size.
    The program will convert to and from pixel coordinates
    as needed for display and for file I/O.
    The program will also convert to and from pixel coordinates
    as needed for OpenCV functions.
    if zoomed in, the program logic should be the same. Only the
    functions that actually draw to the screen should be affected.
    We are still in the normalized coordinate system, normalized
    with respect to the entire frame size. All these "if zoom_rect"
    statements should be removed from mainline code and encapsulated
    within the display functions. This will allow the mainline code to
    express the business logic without carrying the burden of caring
    about the zoomed-in state.

"""

import numpy as np
#import cupy as cp
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import scipy.stats as stats
import cv2 
import os,sys
import queue
import time
import glob
import torch

# local imports
from textbox import put_text_rect

# signature for imported put_text_rect function.
# def put_text_rect(img, text, position, font_face, font_size, rect_color, font_color, font_thickness, tightness=0, anchor='baseline-left'):

# global variables which define the state of the display
displayframe = None
zx0 = None
zy0 = None
zx1 = None
zy1 = None


nmmousex=None
nmmousey=None
pxmousex=None
pxmousey=None


display_mode = 0
DISPLAY_WIDTH = 1920*2//4
DISPLAY_HEIGHT = 1080*2//4

creating_mode = False
hovering_over = None
hovering_over_quad = ""
hires_mode = False
save_flag = False
# drawing transformation state
# transformation state is a 4-tuple of (zx0,zy0,zx1,zy1)
# where x0,y0 is the upper left corner of the rectangle
# and x1,y1 is the lower right corner of the rectangle
#
# the only difference between physical and pixel coordinates
# is the  offset of the zoomed-in area rectangle.

# remember that the zoomed-in area is a rectangle in normalized coordinates
# that is a subset of the entire frame.  The zoomed-in area is then

# this diagram and notes are for the x-axis only.  The y-axis is analogous.

# pixels:      0------------------------------------DISPLAY_WIDTH
# normalized: 0.0------------------------------------1.0
# zoomed:            zx0-----zx1
#              -----/          \---------------------
#             /                                      \
# pixels:     0------------------------------------DISPLAY_WIDTH
# zoom normalized: zx0-----------------------------zx1

# the transformation functions are:

# norm_to_pix = lambda x: int(x*DISPLAY_WIDTH)
# pix_to_norm = lambda x: x/DISPLAY_WIDTH
# when zoomed in:
# zoomed_norm_to_pix = lambda x: int((x-zx0)*DISPLAY_WIDTH/(zx1-zx0))
# zoomed_pix_to_norm = lambda x: x/DISPLAY_WIDTH*(zx1-zx0)+zx0
# the zoomed-in area is a rectangle in normalized coordinates
zx0 = 0
zy0 = 0
zx1 = 1.0
zy1 = 1.0

# examples to verify the correctness of the transformation functions
# print(norm_to_pix(0.5)) # should be DISPLAY_WIDTH//2
# print(pix_to_norm(DISPLAY_WIDTH//2)) # should be 0.5
# print(zoomed_norm_to_pix(0.5)) # should be DISPLAY_WIDTH//2
# print(zoomed_pix_to_norm(DISPLAY_WIDTH//2)) # should be 0.5



def norm2pix(nmrect):
    if len(nmrect) == 4:
        rect = (nmrect[0],nmrect[1],nmrect[0]+nmrect[2],nmrect[1]+nmrect[3])
    else:
        rect=nmrect
    # this should work for points or rectangles,
    # and it will work for x,y,w,h or x0,y0,x1,y1 formats
    if displayframe is None or displayframe.shape[0:2] != (DISPLAY_HEIGHT,DISPLAY_WIDTH):
        print(f"illegal call to pix2norm: displayframe not initialized")
        exit()
    retv = []
    # always take zoomed-in area into account
    for i in range(len(rect)):
        if i % 2 == 0:
            # retv.append((pixrect[i]/DISPLAY_WIDTH*(zx1-zx0)+zx0))
            # break down order of operations of above formula:
            # 1. rect[i]/DISPLAY_WIDTH
            # 2. (result of 1)*(zx1-zx0) 
            # 3. (result of 2)+zx0 
            # now reverse the operations to get the inverse formula:
            retv.append(int(((rect[i]-zx0)/(zx1-zx0))*DISPLAY_WIDTH))
        else:
            retv.append(int(((rect[i]-zy0)/(zy1-zy0))*DISPLAY_HEIGHT))

    if len(nmrect) == 4:
        return retv[0],retv[1],retv[2]-retv[0],retv[3]-retv[1]
    else:
        return tuple(retv)
    return 

frames_to_read = None
frames_to_skip = None


# norm: 0.0--------zx0-nm0----nm1--zx1-------1.0

# pix:  0-----px0---px1---DISPLAY_WIDTH


# px0 formula: px0 = (nm0-zx0)/(zx1-zx0)*DISPLAY_WIDTH
# nm0 formula: nm0 = px0/DISPLAY_WIDTH*(zx1-zx0)+zx0

def pix2norm(pixrect):
    # this should work for points or rectangles,
    # and it will work for x,y,w,h or x0,y0,x1,y1 formats
    if displayframe is None or displayframe.shape[0:2] != (DISPLAY_HEIGHT,DISPLAY_WIDTH):
        print(f"illegal call to pix2norm: displayframe not initialized")
        exit()
    if len(pixrect) == 4:
        rect = (pixrect[0],pixrect[1],pixrect[0]+pixrect[2],pixrect[1]+pixrect[3])
    else:
        rect=list(pixrect)
    retv = []
    # always take zoomed-in area into account
    for i in range(len(pixrect)):
        if i % 2 == 0:
            retv.append((rect[i]/DISPLAY_WIDTH*(zx1-zx0)+zx0))
        else:
            retv.append((rect[i]/DISPLAY_HEIGHT*(zy1-zy0)+zy0))

    if len(pixrect) == 4:
        return retv[0],retv[1],retv[2]-retv[0],retv[3]-retv[1]
    else:
        return tuple(retv)


def norm_to_imgpix(img,rect):
    retv = []
    for i in range(len(rect)):
        if i % 2 == 0:
            retv.append(int(rect[i]*img.shape[1]))
        else:
            retv.append(int(rect[i]*img.shape[0]))
    return tuple(retv)

def imgpix_to_norm(img,rect):
    retv = []
    for i in range(len(rect)):
        if i % 2 == 0:
            retv.append(rect[i]/img.shape[1])
        else:
            retv.append(rect[i]/img.shape[0])
    return tuple(retv)


'''
def rect_transp_fill(img,rect,color, alpha=0.5, border=-1):
    # This is a low-level function that only operates on pixel coordinates.  
    #   The caller is responsible for any conversions.
    imgcopy = img.copy()
    x0, y0, x1, y1 =  (rect[0], rect[1], rect[2], rect[3])
    rx0 = x0+1
    ry0 = y0+1
    rx1 = x1-1
    ry1 = y1-1
    cv2.rectangle(imgcopy,(rx0,ry0),(rx1,ry1),color,border)
    cv2.addWeighted(imgcopy, alpha, img, 1-alpha, 0, img)
    return imgcopy
'''


STREAKING=False
NUM_MED = 3
FRAME_MEDIAN_STRIDE = 3
def sigmoid(x):
    return 1/(1+np.exp(-(x-0.5)*10))







def read_movie(uncfi):
    cap = None
    cap = cv2.VideoCapture(uncfi)
    if cap is None:
        print(f"opening video file {uncfi} failed.")
        exit()
    frames = []
    ok,frame=cap.read()
    while ok:
        frames.append(frame) #cv2.GaussianBlur(frame, (1,1),0))
        print(f"captured {len(frames)} frames...")
        if len(frames) > frames_to_read:
            break
        ok,frame=cap.read()
    cap.release()
    del cap
    cap=None
    return frames

# convert rectangle 
def rect2yolo(rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    cx = x+w/2
    cy = y+h/2
    return (cx,cy,w,h)

def yolo2rect(yolo):
    cx = yolo[0]
    cy = yolo[1]
    w  = yolo[2]
    h  = yolo[3]

    x = cx-w/2
    y = cy-h/2
    return (x,y,w,h)


fdf_filenames = sorted(glob.glob("../corrected/*.mp4"))
print (f"found {len(fdf_filenames)} files")

#fdf = pd.read_csv("oflowavg.csv", header=None)
#fdf.columns = ["filename","ofmean","ofmin","ofmax","ofstd","N"]

#fdf.sort_values(by="filename", ascending=True, inplace=True,ignore_index=True)
fdf_uncorrected = [ fi.replace("/corrected/","/uncorrected/").replace("_corrected","") for fi in fdf_filenames ]
fdf_labels = [ fi.replace("/corrected/","/labels/").replace("_corrected","").replace(".mp4",".csv") for fi in fdf_filenames ]

# MOUSE EVENT QUEUE
mouseq = queue.Queue()
def mouse_event(e, x,y,flags,param):
    # print(f"mouse_event({e},{x},{y},{flags},{param})")
    # always return raw mouse coordinates ~= pixel coordinates
    mouseq.put_nowait({"e":e, "x":x, "y":y, "flags":flags, "param":param})
    return True

# END MOUSE EVENT QUEUE
window_setup = False
findex = 0
if os.path.exists("findex.txt"):
    findex = int(open("findex.txt").read())
while True:
    frames_to_read = 150
    frames_to_skip = 1
    if hires_mode:
        frames_to_read = 999
        frames_to_skip =3 
    if findex < 0:
        findex = 0
    if findex >= len(fdf_filenames):
        findex = len(fdf_filenames)-1
    # save findex
    open("findex.txt","w").write(f"{findex}")
    cfi =   fdf_filenames[findex]
    uncfi = fdf_uncorrected[findex]
    labelsfi = fdf_labels[findex]

    # for each file,
    print(f"cfi={cfi}")
    # file name format: ../corrected/SLBE_20230831_155801_corrected.mp4
    frames = read_movie(cfi)
    ucframes = read_movie(uncfi)
    print("done reading movies")
    annotations = {}
    annotdf = None
    if os.path.exists(labelsfi):
        try:
            annotdf = pd.read_csv(labelsfi,header=None,sep=' ')
        except pd.errors.EmptyDataError:
            annotdf = None
    if annotdf is not None:
        annotdf.columns = ("frame class cx cy w h".split(" "))
        print (annotdf)
        #annotdf["frame"] //= frames_to_skip
        for rowi in range(annotdf.shape[0]):
            framei = annotdf.loc[rowi,"frame"]
            print(f"frame={framei}")
            yoloroi = annotdf.loc[rowi,["cx","cy","w","h"]].to_list()
            rectroi = list(yolo2rect(yoloroi))
            if framei in range(len(annotations)):
                annotations[framei].append(rectroi)
            else:
                annotations[framei] = [ rectroi ]
    print(f"annotations: {annotations}")
    print("converting frames to numpy float32")
    frames = np.array(frames,dtype='float32')
    print("normalizing frames")
    frames = np.float32(frames) / np.float32(frames.max())
    print("done normalizing frames")
    nframes = frames.shape[0]
    #fmean = frames.mean()
    #for i in range(frames.shape[0]):
    #    frames[i,...] = frames[i,...] * (fmean/frames[i,...].mean())
    print(f"frames.shape={frames.shape}")
    print("copying frames to frames_org")
    frames_org = frames.copy()
    mdiff = np.zeros((nframes,frames.shape[1],frames.shape[2]),dtype='float32')
    print(f"frames.shape={frames.shape}")

    meds = []
    print(f"copying {nframes//FRAME_MEDIAN_STRIDE} frames to GPU")
    framescu = np.array(frames)
    starttime = time.time()
    for i in range(0,nframes,FRAME_MEDIAN_STRIDE):
        print(f"median {i}...",end='',flush=True)
        rangelo=np.clip(i-NUM_MED,0,nframes-1)
        rangehi=np.clip(i+NUM_MED,0,nframes-1)
        meds.append(torch.median(torch.tensor(framescu[rangelo:rangehi,...,1]),dim=0).values.numpy())
        #meds.append(np.median(framescu[rangelo:rangehi,...,1],axis=0))
        #meds.append(np.median(frames[rangelo:rangehi,...,1],axis=0))
        print("done",flush=True)
    # interpolate medians
    imeds = []
    for j in range(0,nframes):
        if j % FRAME_MEDIAN_STRIDE == 0:
            imeds.append(meds[j//FRAME_MEDIAN_STRIDE])
        else:
            if j > nframes//FRAME_MEDIAN_STRIDE:
                imeds.append(meds[j//FRAME_MEDIAN_STRIDE])
            else:
                r = (j % FRAME_MEDIAN_STRIDE) / FRAME_MEDIAN_STRIDE
                imeds.append(
                    meds[j//FRAME_MEDIAN_STRIDE] * (1.0-r) +
                    meds[j//FRAME_MEDIAN_STRIDE+1] * r
                    )
    print(f"median time: {time.time()-starttime}")

    for i in range(frames.shape[0]):

        mdif = cv2.absdiff(imeds[i],frames[i,...,1])
        #mdif = (cv2.GaussianBlur(mdif, (9,9),6))
        print(f"mdif min={mdif.min()}, max={mdif.max()}")
        mdiff[i,...] = mdif

    '''
    mdif2=mdiff.copy()
    for i in range(1, frames.shape[0]):
        mdif2[i,...] = np.float32(cv2.absdiff(mdiff[i-1,...],mdiff[i,...])).copy()
    '''    
    print(f"collecting mdiffs")
    mdiff = mdiff / mdiff.max()
    #mdif2 = mdif2 / mdif2.max()
    print(f"done mdiffs.")
    if STREAKING:
        print(f"streaking...")
        canvas = np.ones((mdiff.shape[1],mdiff.shape[2],3),dtype='float32')*frames[0]*0.3
        canvas[...,0]=canvas[...,1]
        canvas[...,1] *= 0
        canvas[...,2] *= 0
        streaks = [canvas[...,1]]
        for i in range(1,mdiff.shape[0]):
            dif = mdiff[i,...] #np.clip(mdiff[i,...]-mdiff[i-1,...],0,9999)
            dif = sigmoid((dif-0.4)*4)
            #difquant = np.quantile(dif,0.9999)
            #dif[dif<difquant] = 0
            #dif[dif>0] = 1
            #dif = dif/dif.max()

            hsv = np.zeros((mdiff.shape[1],mdiff.shape[2],3),dtype='uint8')
            hsv[...,0] += np.uint8(135 - (i/mdiff.shape[0])*110.0)
            hsv[...,1] += 255
            hsv[...,2] = np.uint8(dif*255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bgr = np.float32(bgr/bgr.max())
            dif3 = np.zeros((mdiff.shape[1],mdiff.shape[2],3),dtype='float32')
            dif3[...,0]=dif
            dif3[...,1]=dif
            dif3[...,2]=dif
            # blending formula
            #canvas = canvas*(1-dif3)+dif3*bgr
            canvas = np.clip(canvas+bgr,0,1)
            streaks.append(canvas/canvas.max())
        print(f"blob detection setup...")
        #mdiff = cv2.absdiff(mdiff,frames[NUM_MED:-NUM_MED,...,1])
        #mdiffq9 = np.quantile(mdiff,.959)
        #mdiff[mdiff<mdiffq9] = 0
        #mdiff[mdiff>0] = 1
        params = cv2.SimpleBlobDetector_Params()
        print(f"params object created, setting values.")
        params.filterByArea = True;
        params.minArea = 100
        params.maxArea = 9999
        params.minThreshold = 0
        params.maxThreshold = 70
        params.filterByCircularity=False
        params.filterByInertia=True
        params.minInertiaRatio=0.001
        params.maxInertiaRatio=0.9
        params.filterByConvexity=False
        print(f"creating detector.")
        detector = cv2.SimpleBlobDetector_create(params)
        print(f"normalizing mdiff.")
        mdiff=mdiff/mdiff.max()
        #print(f"sigmoid(sigmoid()) calc")
        #mdiff = sigmoid(sigmoid(mdiff*1.5))
        blobKeypoints = []
        for i in range(mdiff.shape[0]):
            print(f"blob detection {i}...")
            mint = np.uint8(mdiff[i]/mdiff[i].max()*255)
            kpts = detector.detect(np.clip(255-mint,0,255))
            blobKeypoints.append(kpts)
    #mmean = mdiff.mean()
    #for i in range(mdiff.shape[0]):
    #    mdiff[i,...] = mdiff[i,...] * mmean / mdiff[i,...].mean()

    #frames = frames[NUM_MED:-NUM_MED,...]
    print(f"normalizing frames")
    #frames = frames/frames.max()
    print(f"blending frames and mdiffs")
    frames[...,0] = frames[...,0]*(mdiff*2+0.3)
    frames[...,1] = frames[...,1]*(mdiff*2+0.3)
    frames[...,2] = frames[...,2]*mdiff*4 + mdiff*0.5

    #frames=mdiff#frames[...,1]
    finish = False
    imgs = []
    deleting_mode = False
    deleting_ann = None
    rx0=None
    ry0=None
    rx1=None
    ry1=None
    dragging_mode = False
    dragging_origin_x = None
    dragging_origin_y = None
    dragging_ai = None

    pause_mode = True
    one_shot = False
    pos_frames = 0
    if os.path.exists("posframes.txt"):
        pos_frames = int(open("posframes.txt").read())
    pos_frames = np.clip(pos_frames,0,frames.shape[0]-1)
    ok = True
    if STREAKING:
        num_display_modes = 6
    else:
        num_display_modes = 5
    while ok and not finish:
        flash = (int(time.time()*3)%2) > 0
        if window_setup==False:
            print(f"setting up GUI")
            cv2.namedWindow('frame' , cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('frame', DISPLAY_WIDTH, DISPLAY_HEIGHT+64)
            cv2.moveWindow('frame', 150, 15)
            cv2.setMouseCallback('frame',mouse_event)
            window_setup = True
        #print(f"pos_frames={pos_frames}")
        #print(f"len(ucframes)={len(ucframes)}")
        if display_mode == 0:
            predisplayframe = ucframes[pos_frames].copy()
        if display_mode == 1:
            predisplayframe = frames_org[pos_frames].copy()
        elif display_mode == 2:
            predisplayframe = frames[pos_frames].copy()
        elif display_mode == 3:
            predisplayframe = mdiff[pos_frames].copy()
        elif display_mode == 4:
            # optical flow
            # always compute optical flow from the original frames
            cur = cv2.cvtColor(frames_org[pos_frames],cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frames_org[pos_frames+1],cv2.COLOR_BGR2GRAY)

            flow0 = cv2.calcOpticalFlowFarneback(cur,nxt,None,0.5,3,3,3,5,1.2,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = cv2.calcOpticalFlowFarneback(cur,nxt,flow0,0.5,3,3,3,5,1.2,cv2.OPTFLOW_USE_INITIAL_FLOW) #cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flowmag = np.sqrt(flow[...,0]**2+flow[...,1]**2)
            #flowmag = np.log(np.log(flowmag+1)+1)
            flowmag50 = np.quantile(flowmag,0.90)
            flowmag[flowmag>flowmag50] = flowmag50
            if hovering_over is not None and pos_frames in annotations:
                an = annotations[pos_frames][hovering_over]
                pxls = norm_to_imgpix(an)
                flow = flow[pxls[1]:(pxls[1]+pxls[3]),pxls[0]:(pxls[0]+pxls[2])]
                flowmag = np.sqrt(flow[...,0]**2+flow[...,1]**2)
                flowmag = flowmag / flowmag.max()
                predisplayframe = np.zeros_like(frames[pos_frames])
                predisplayframe[pxls[1]:(pxls[1]+pxls[3]),pxls[0]:(pxls[0]+pxls[2]),0] = np.stack([flowmag]*3,axis=-1)
            else:
                print("flowmag max=",flowmag.max())
                flowmag = flowmag / flowmag.max()
                predisplayframe = np.stack([flowmag]*3,axis=-1)
        elif display_mode == 5:
            predisplayframe = streaks[pos_frames].copy()

        phyz = norm_to_imgpix(predisplayframe,(zx0,zy0,zx1,zy1))
        predisplayframe = predisplayframe[phyz[1]:phyz[3],phyz[0]:phyz[2],...]
        displayframe = cv2.resize(predisplayframe,(DISPLAY_WIDTH,DISPLAY_HEIGHT))
        del predisplayframe 

        # IMPORTANT: Pixel coordinates are now good on the displayframe image. 
        #
        # Any calls to norm2pix/pix2norm before this point are invalid.
        # no need to rescale zoomed-in area, since it is already in pixel coordinates.
        # no need to rescale annotations, since they are in normalized coordinates.
        # no need to rescale mouse coordinates, since they are in pixel coordinates.
        # no need to rescale dragging_origin, since it is in normalized coordinates
        # the only time we move out of normalized coordinates is when drawing to displayframe.
    

        hovering_over = None
        hovering_over_quad = ""
        if pos_frames in annotations: # if there are annotations for this frame
            for ai in range(len(annotations[pos_frames])):
                an = annotations[pos_frames][ai]
                pxan = norm2pix(an)
                # print(f"an={an}  pxan={pxan}")
                cv2.rectangle(displayframe,pxan,(0,0,255),1)
                put_text_rect(displayframe,f"{ai}",(pxan[0],pxan[1]),cv2.FONT_HERSHEY_DUPLEX,0.35,(255,255,0),(0,0,0),1,anchor="top-right")
      
                # see if we are hovering over this annotation
                #        ex[0]  
                #  ex[1]  +-----------------+ 
                #         |                 |
                #         |                 |
                #         +-----------------+ ex[1]+ex[3]
                #         ex[0]+ex[2]
                #
                if nmmousex is not None and nmmousex >= an[0] and nmmousex <= an[0]+an[2] and nmmousey >= an[1] and nmmousey <= an[1]+an[3]:
                    cv2.rectangle(displayframe,norm2pix(an),(255,220,220),1)
                    if dragging_mode:
                        cv2.rectangle(displayframe,norm2pix(an),(255,255,127),1)
                    hovering_over = ai
                    # determine which diagonal-delimited quadrant the mouse is in
                    # these calculations are in normalized coordinates
                    CROSS_COLOR = (0,50,255)
                    rect_center_x = an[0]+an[2]/2
                    rect_center_y = an[1]+an[3]/2
                    min_width = min(an[2],an[3])/4
                    dist_from_center_x = nmmousex - rect_center_x
                    dist_from_center_y = nmmousey - rect_center_y
                    dist_from_center = np.sqrt(dist_from_center_x**2 + dist_from_center_y**2)
                    px = norm2pix(an)
                    if dist_from_center < min_width:
                        cv2.drawMarker(displayframe,norm2pix((an[0]+an[2]/2,an[1]+an[3]/2)),CROSS_COLOR,cv2.MARKER_CROSS,10,2)
                        hovering_over_quad = "C"
                    elif nmmousex < an[0]+an[2]/2:
                        if nmmousey < an[1]+an[3]/2:
                            cv2.drawMarker(displayframe,(px[0],px[1]),CROSS_COLOR,cv2.MARKER_CROSS,10,2)
                            hovering_over_quad = "NW"
                        else:
                            cv2.drawMarker(displayframe,(px[0],px[1]+px[3]-1),CROSS_COLOR,cv2.MARKER_CROSS,10,2)
                            hovering_over_quad = "SW"
                    else: # nmmousex >= ex[0]+ex[2]//2
                        if nmmousey < an[1]+an[3]/2:
                            cv2.drawMarker(displayframe,(px[0]+px[2]-1,px[1]),CROSS_COLOR,cv2.MARKER_CROSS,10,2)
                            hovering_over_quad = "NE"
                        else:
                            cv2.drawMarker(displayframe,(px[0]+px[2]-1,px[1]+px[3]-1),CROSS_COLOR,cv2.MARKER_CROSS,10,2)
                            hovering_over_quad = "SE"
                    if deleting_mode:
                        if flash:
                            put_text_rect(displayframe,"delete(y/n)",(px[0],px[1]),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,127,255),(0,0,0),1, anchor="bottom-left")
                    else:
                        #old: put_text_rect(displayframe,f"hov: {pos_frames}:{hovering_over}{hovering_over_quad}",(ex[0],ex[1]),(255,255,255),cv2.FONT_HERSHEY_DUPLEX,0.5,1)
                        put_text_rect(displayframe,
                                f"hov: {pos_frames}:{hovering_over}{hovering_over_quad}",
                                norm2pix((an[0],an[1])),
                                cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),(0,0,0),1,
                                anchor="bottom-left")
        cv2.putText(displayframe,f"{findex:04d}:{labelsfi}:{pos_frames:03d}:{deleting_mode}:{deleting_ann}",(10,30),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        # crosshair
        if pxmousex is not None and pxmousey is not None:
            displayframe[pxmousey,...] =  1.0 - displayframe[pxmousey,...]
            displayframe[:,pxmousex,...] = 1.0 - displayframe[:,pxmousex,...]
        cv2.imshow('frame', displayframe)

        one_pixel = (zx1-zx0)/DISPLAY_WIDTH

        # print(f"frame redrawn: pos_frames={pos_frames}")
        # check for keyboard events
        k = cv2.waitKeyEx(1) # & 0xFF
        if (k != 255 and k != -1):
            print(f"k={k}")
        if k == ord('q'):
            finish = True
            break
        elif k == ord('h'):
            hires_mode = True
            break
        elif k == ord('l'):
            hires_mode = False
            break
        elif k == ord('n'):
            if deleting_mode:
                deleting_mode = False
                deleting_ann = None
                print('deleting cancelled')
            else:
                findex += 1
                hires_mode = False
                findex = findex % len(fdf_filenames)
                finish=False
                break;
        elif k == ord('p'):
            if findex > 0:
                findex -= 1
                hires_mode = False
                break
        elif k >= ord('0') and k <= ord('9'):
            findex = len(fdf_filenames) * (k-ord('0')) // 10
            finish = False
            break

        elif k == 27:  # escape key
            zx0 = 0
            zy0 = 0
            zx1 = 1.0
            zy1 = 1.0
        elif k == ord(' '):
            pause_mode = not pause_mode
        elif k == ord(','):
            pause_mode = True
            if pos_frames > 0:
                pos_frames -= 1
        elif k == ord('.'):
            pause_mode = True
            if pos_frames < len(frames)-2:
                pos_frames += 1
        elif k == 10 or k == 13:
            display_mode = (display_mode + 1) % num_display_modes
        elif k == ord('o'):
            if pos_frames+1 < frames.shape[0] and hovering_over is not None and pos_frames in annotations:
                print(f"applying optical flow to rectangle {hovering_over}")
                cur = frames_org[pos_frames][...,1]
                nxt = frames_org[pos_frames+1][...,1]
                maxall = np.max([cur.max(),nxt.max()])

                #cur = cv2.cvtColor(frames[pos_frames],cv2.COLOR_BGR2GRAY)
                #nxt = cv2.cvtColor(frames[pos_frames+1],cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(cur,nxt,None,0.5,3,1,3,5,1.2,0) #cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                flowmag = np.sqrt(flow[...,0]**2+flow[...,1]**2)
                an = annotations[pos_frames][hovering_over]
                pyx = norm_to_imgpix(flow,an)
                flow = flow[int(pyx[1]):int(pyx[1]+pyx[3]),int(pyx[0]):int(pyx[0]+pyx[2])]
                fs = flow.shape
                #flow=flow[fs[0]//4:fs[0]*3//4,fs[1]//4:fs[1]*3//4]
                # compute mean of flow over rectangle
                #flowx = np.mean(flow[...,0]) * DISPLAY_WIDTH
                #flowy = np.mean(flow[...,1]) * DISPLAY_HEIGHT
                flowx = np.quantile(flow[...,0],0.990)
                flowy = np.quantile(flow[...,1],0.990)
                print(f"flowx={flowx}, flowy={flowy}")
                # apply flow to rectangle
                newan = an.copy()
                newan[0] += flowx
                newan[1] += flowy
                if pos_frames+1 in annotations:
                    annotations[pos_frames+1].append(newan)
                else:
                    annotations[pos_frames+1] = [ newan ]
                pos_frames += 1
                save_flag = True


        elif k == ord('x'):
            if hovering_over is not None and not deleting_mode:
                deleting_mode = True
                deleting_ann = hovering_over
                print(f"deleting mode on")
            else:
                deleting_mode = False
                deleting_ann = None
                print('deleting cancelled')

        elif k == ord('y'):
            if deleting_mode:
                deleting_mode = False
                # 'y' has been pressed following an 'x', delete the annotation
                # delete the annotation under the mouse
                if deleting_ann is not None and pos_frames in annotations:
                    del annotations[pos_frames][deleting_ann]
                    #deleting_ann = None
                    # write to file
                    # must rewrite the entire file
                    ww=frames.shape[2]
                    wh=frames.shape[1]
                    open(labelsfi,"w").write("")
                    for p_frames in annotations:
                        for ai in range(len(annotations[p_frames])):
                            if p_frames == pos_frames and ai == deleting_ann:
                                continue
                            an = annotations[p_frames][ai]
                            yoloroi = list(rect2yolo(an))
                            open(labelsfi,"a").write(f"{p_frames} 0 {yoloroi[0]} {yoloroi[1]} {yoloroi[2]} {yoloroi[3]}\n")
                    # save pos_frames
                    open("posframes.txt","w").write(f"{pos_frames}")
                    deleting_ann = None
                    pause_mode = True

        elif (k == 81 or k == 65361 or k == ord('a')) and hovering_over is not None: # left arrow
            if hovering_over_quad == "C":
                annotations[pos_frames][hovering_over][0] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][0] -= one_pixel
                annotations[pos_frames][hovering_over][2] += one_pixel
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][2] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][0] -= one_pixel
                annotations[pos_frames][hovering_over][2] += one_pixel
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][2] -= one_pixel
                save_flag = True
        elif (k == 83 or k == 65363 or k == ord('d')) and hovering_over is not None: # right arrow
            if hovering_over_quad == "C":
                annotations[pos_frames][hovering_over][0] += one_pixel
                save_flag = True
            elif hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][0] += one_pixel
                annotations[pos_frames][hovering_over][2] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][2] += one_pixel
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][0] += one_pixel
                annotations[pos_frames][hovering_over][2] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][2] += one_pixel
                save_flag = True
        elif (k == 84 or k == 65364 or k == ord('s')) and hovering_over is not None: # down arrow
            if hovering_over_quad == "C":
                annotations[pos_frames][hovering_over][1] += one_pixel
                save_flag = True
            elif hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][1] += one_pixel
                annotations[pos_frames][hovering_over][3] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][1] += one_pixel
                annotations[pos_frames][hovering_over][3] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][3] += one_pixel
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][3] += one_pixel
                save_flag = True
        elif (k == 82 or k == 65362 or k == ord('w')) and hovering_over is not None: # up arrow
            if hovering_over_quad == "C":
                annotations[pos_frames][hovering_over][1] -= one_pixel
                save_flag = True
            if hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][1] -= one_pixel
                annotations[pos_frames][hovering_over][3] += one_pixel
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][1] -= one_pixel
                annotations[pos_frames][hovering_over][3] += one_pixel
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][3] -= one_pixel
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][3] -= one_pixel
                save_flag = True
            


        if save_flag:
            open(labelsfi,"w").write("")
            for p_frames in annotations:
                for ai in range(len(annotations[p_frames])):
                    an = annotations[p_frames][ai]
                    yoloroi = list(rect2yolo(an))
                    open(labelsfi,"a").write(f"{p_frames} 0 {yoloroi[0]} {yoloroi[1]} {yoloroi[2]} {yoloroi[3]}\n")
            save_flag = False


        mevent = None
        try:
            mevent = mouseq.get_nowait()
        except queue.Empty:
            mevent = None

        if mevent is not None:
            event = mevent['e']
            rawmousex = mevent['x']
            rawmousey = mevent['y']
            nmmousex,nmmousey = pix2norm((rawmousex,rawmousey))
            pxmousex,pxmousey = ((rawmousex,rawmousey))
            print(f"mouse event = {event} ({rawmousex:.04f},{rawmousey:0.04f}) ({nmmousex:.04f},{nmmousey:.04f}) ({pxmousex},{pxmousey})")
            if event==cv2.EVENT_MBUTTONDOWN:
                # poor man's zoom until we move to Qt or whatever.
                print(f"clicked.")
                # calculate new zoom level in normalized coordinates
                nmrectwidth = zx1-zx0
                nmrectheight =zy1-zy0
                if nmrectwidth < 5/DISPLAY_WIDTH or nmrectheight < 5/DISPLAY_HEIGHT:
                    zx0 = 0
                    zy0 = 0
                    zx1 = 1.0
                    zy1 = 1.0
                    print(f"zoomed out")
                else:
                    nmrectx = zx0 + nmmousex*nmrectwidth/2
                    nmrecty = zy0 + nmmousey*nmrectheight/2
                    nmrectwidth /= 2
                    nmrectheight /= 2
                    zx0 = nmrectx
                    zy0 = nmrecty
                    zx1 = nmrectx + nmrectwidth
                    zy1 = nmrecty + nmrectheight
                print(f"zoomed to (zx0:{zx0:.03f},zy0:{zy0:.03f}),(zx1:{zx1:.03f},zy1:{zy1:.03f})")
            elif event == cv2.EVENT_MOUSEMOVE:
                if creating_mode:
                    #if we've dragged more than 5 pixels, create a new annotation
                    print(f"dragging ({dragging_origin_x},{dragging_origin_y}),({nmmousex},{nmmousey})")
                    #cv2.line(displayframe,(dragging_origin_x,dragging_origin_y),(mx,my),(255,255,255),1)
                    # if we've dragged more than 20 pixels, create a new annotation
                    # must first convert both dragging_origin to pixel coordinates
                    pxdx,pxdy = norm2pix((dragging_origin_x,dragging_origin_y))
                    if abs(pxmousex-pxdx) > 20 or abs(pxmousey-pxdy) > 20:
                        # establish the new annotation rectangle
                        # different rules depending on what direction we dragged
                        if nmmousex < dragging_origin_x:
                            if nmmousey < dragging_origin_y:
                                newan = [nmmousex,nmmousey,dragging_origin_x-nmmousex,dragging_origin_y-nmousey]
                            else:
                                newan = [nmmousex,dragging_origin_y,dragging_origin_x-nmmousex,nmmousey-dragging_origin_y]
                        else:
                            if nmmousey < dragging_origin_y:
                                newan = [dragging_origin_x,nmmousey,nmmousex-dragging_origin_x,dragging_origin_y-nmmousey]
                            else:
                                newan = [dragging_origin_x,dragging_origin_y,nmmousex-dragging_origin_x,nmmousey-dragging_origin_y]
                        if pos_frames in annotations:
                            annotations[pos_frames].append(newan)
                        else:
                            annotations[pos_frames] = [ newan ]
                        save_flag = True
                        creating_mode = False
                        dragging_mode = True
                        dragging_ai = len(annotations[pos_frames])-1
                        dragging_origin_x = nmmousex
                        dragging_origin_y = nmmousey

                if dragging_mode:
                    if dragging_ai is not None:
                        if hovering_over_quad == "C":
                            annotations[pos_frames][dragging_ai][0] += nmmousex-dragging_origin_x
                            annotations[pos_frames][dragging_ai][1] += nmmousey-dragging_origin_y
                        elif hovering_over_quad == "NW":
                            annotations[pos_frames][dragging_ai][0] += nmmousex-dragging_origin_x
                            annotations[pos_frames][dragging_ai][1] += nmmousey-dragging_origin_y
                            annotations[pos_frames][dragging_ai][2] -= nmmousex-dragging_origin_x
                            annotations[pos_frames][dragging_ai][3] -= nmmousey-dragging_origin_y
                        elif hovering_over_quad == "NE":
                            annotations[pos_frames][dragging_ai][2] += nmmousex-dragging_origin_x
                            annotations[pos_frames][dragging_ai][1] += nmmousey-dragging_origin_y
                            annotations[pos_frames][dragging_ai][3] -= nmmousey-dragging_origin_y
                        elif hovering_over_quad == "SW":
                            annotations[pos_frames][dragging_ai][3] += nmmousey-dragging_origin_y
                            annotations[pos_frames][dragging_ai][0] += nmmousex-dragging_origin_x
                            annotations[pos_frames][dragging_ai][2] -= nmmousex-dragging_origin_x
                        elif hovering_over_quad == "SE":
                            annotations[pos_frames][dragging_ai][2] += nmmousex-dragging_origin_x
                            annotations[pos_frames][dragging_ai][3] += nmmousey-dragging_origin_y
                        annotations[pos_frames][dragging_ai][0] = np.clip(annotations[pos_frames][dragging_ai][0],0,1)
                        annotations[pos_frames][dragging_ai][1] = np.clip(annotations[pos_frames][dragging_ai][1],0,1)
                        annotations[pos_frames][dragging_ai][2] = np.clip(annotations[pos_frames][dragging_ai][2],0,1)
                        annotations[pos_frames][dragging_ai][3] = np.clip(annotations[pos_frames][dragging_ai][3],0,1)

                        dragging_origin_x,dragging_origin_y = (nmmousex,nmmousey)
                        save_flag = True

            elif event == cv2.EVENT_LBUTTONDOWN:
                # if we are hovering over an annotation, start dragging it
                if hovering_over is not None and hovering_over >= 0 and hovering_over < len(annotations[pos_frames]):
                    dragging_mode = True
                    dragging_ai = hovering_over
                    dragging_origin_x = nmmousex
                    dragging_origin_y = nmmousey
                else:
                    # create a new annotation
                    # clear out any dragging mode info
                    # and replace with current needed values.
                    dragging_mode = False
                    dragging_ai = None
                    dragging_origin_x,dragging_origin_y = nmmousex,nmmousey
                    creating_mode = True

            elif event == cv2.EVENT_LBUTTONUP:
                dragging_mode = False
                creating_mode = False
                dragging_ai = None

        if not pause_mode:
            pos_frames += 1
            if pos_frames >= len(frames):
                pos_frames = 0
        if pause_mode:
            time.sleep(0.00005)
        #if zoom_rect:
        #    time.sleep(0.05);
    # done with this video
    # if we are done entirely, break
    if finish:
        break
    
cv2.destroyAllWindows()

