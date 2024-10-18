#!/usr/bin/env python3

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
mx=-1
my=-1
display_mode = 0
DISPLAY_WIDTH = 1920*2//4
DISPLAY_HEIGHT = 1080*2//4

hovering_over = None
hovering_over_quad = ""
hires_mode = False
size_ratio_x = 1.0
size_ratio_y = 1.0
frames_to_read = None
frames_to_skip = None

def rect_transp_fill(img,rect,color, alpha=0.5, border=-1):
    imgcopy = img.copy()
    x0, y0, x1, y1 =  [rect[0], rect[1], rect[2], rect[3]]
    rx0 = x0+1
    ry0 = y0+1
    rx1 = x1-1
    ry1 = y1-1
    cv2.rectangle(imgcopy,(rx0,ry0),(rx1,ry1),color,border)
    cv2.addWeighted(imgcopy, alpha, img, 1-alpha, 0, img)
    return imgcopy


STREAKING=False
NUM_MED = 3
FRAME_MEDIAN_STRIDE = 1
def sigmoid(x):
    return 1/(1+np.exp(-(x-0.5)*10))



frames_to_read = 50
frames_to_skip = 5
if hires_mode:
    frames_to_read = 999
    frames_to_skip =1 




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

def rect2yolo(rect,ww,wh):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    cx = x+w//2
    cy = y+h//2
    return (cx/ww,cy/wh,w/ww,h/wh)

def yolo2rect(yolo,ww,wh):
    cx = int(yolo[0] * ww)
    cy = int(yolo[1] * wh)
    w = int(np.abs(yolo[2]*ww))
    h = int(np.abs(yolo[3]*wh))

    x = cx-w//2
    y = cy-h//2
    return (x,y,w,h)


def select_roi(frame):

    roi = cv2.selectROI(frame)
    return roi

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
    mouseq.put_nowait({"e":e, "x":x*size_ratio_x, "y":y*size_ratio_y, "flags":flags, "param":param})
# END MOUSE EVENT QUEUE
window_setup = False
findex = 0
if os.path.exists("findex.txt"):
    findex = int(open("findex.txt").read())
while True:
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
        ww = frames[0].shape[1]
        wh = frames[0].shape[0]
        for rowi in range(annotdf.shape[0]):
            framei = annotdf.loc[rowi,"frame"]
            print(f"frame={framei}")
            yoloroi = annotdf.loc[rowi,["cx","cy","w","h"]].to_list()
            rectroi = list(yolo2rect(yoloroi,ww,wh))
            if framei in annotations:
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
    pause_mode = True
    one_shot = False
    pos_frames = 0
    if os.path.exists("posframes.txt"):
        pos_frames = int(open("posframes.txt").read())
    ok = True
    if STREAKING:
        num_display_modes = 6
    else:
        num_display_modes = 5
    pause_phase = 0
    zoom_rect = None
    editing_rect = False
    ex0=None
    ey0=None
    ex1=None
    ey1=None
    mdblclk=False



    while ok:
        frames_to_read = 50
        frames_to_skip = 5
        if hires_mode:
            frames_to_read = 999
            frames_to_skip =1 
        if not zoom_rect:
            zx0 = 0
            zy0 = 0
            zx1 = frames.shape[2]
            zy1 = frames.shape[1]
        if pause_mode:
            pause_phase += np.pi/90.
        else:
            pause_phase = 0
        #canvasframe = frames[pos_frames]*(np.cos(pause_phase)**2) + frames_org[pos_frames] * (1-np.cos(pause_phase)**2)
        #canvasframe = np.uint8(255-mdiff[pos_frames]*255)
        #canvasframe = np.stack([canvasframe]*3,axis=-1)
        #canvasframe = cv2.drawKeypoints(canvasframe, blobKeypoints[pos_frames],(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #print(blobKeypoints[pos_frames])
        if window_setup==False:
            print(f"setting up GUI")
            cv2.namedWindow('frame') # , cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', DISPLAY_WIDTH, DISPLAY_HEIGHT)
            cv2.moveWindow('frame', 150, 5)
            cv2.setMouseCallback('frame',mouse_event)
            window_setup = True
        #print(f"pos_frames={pos_frames}")
        #print(f"len(ucframes)={len(ucframes)}")
        if display_mode == 0:
            displayframe = ucframes[pos_frames].copy()
        if display_mode == 1:
            displayframe = frames_org[pos_frames].copy()
        elif display_mode == 2:
            displayframe = frames[pos_frames].copy()
        elif display_mode == 3:
            displayframe = mdiff[pos_frames].copy()
        elif display_mode == 4:
            # optical flow
            # always compute optical flow from the original frames
            cur = cv2.cvtColor(frames[pos_frames],cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frames[pos_frames+1],cv2.COLOR_BGR2GRAY)
            flow0 = cv2.calcOpticalFlowFarneback(cur,nxt,None,0.5,3,3,3,5,1.2,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = cv2.calcOpticalFlowFarneback(cur,nxt,flow0,0.5,3,3,3,5,1.2,cv2.OPTFLOW_USE_INITIAL_FLOW) #cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flowmag = np.sqrt(flow[...,0]**2+flow[...,1]**2)
            #flowmag = np.log(np.log(flowmag+1)+1)
            flowmag50 = np.quantile(flowmag,0.90)
            flowmag[flowmag>flowmag50] = flowmag50
            if hovering_over is not None and pos_frames in annotations:
                an = annotations[pos_frames][hovering_over]
                flow = flow[int(an[1]):int(an[1]+an[3]),int(an[0]):int(an[0]+an[2])]
                flowmag = np.sqrt(flow[...,0]**2+flow[...,1]**2)
                flowmag = flowmag / flowmag.max()
                displayframe = np.zeros_like(frames[pos_frames])
                displayframe[an[1]:an[1]+an[3],an[0]:an[0]+an[2]] = np.stack([flowmag]*3,axis=-1)
            else:
                print("flowmag max=",flowmag.max())
                flowmag = flowmag / flowmag.max()
                displayframe = np.stack([flowmag]*3,axis=-1)
        elif display_mode == 5:
            displayframe = streaks[pos_frames].copy()
        if zoom_rect:
            displayframe = displayframe[zy0:zy1,zx0:zx1,...]


        phy_size = displayframe.shape[1],displayframe.shape[0]
        displayframe = cv2.resize(displayframe,(DISPLAY_WIDTH,DISPLAY_HEIGHT))
        size_ratio_x = phy_size[0] / DISPLAY_WIDTH
        size_ratio_y = phy_size[1] / DISPLAY_HEIGHT 
        hovering_over = None
        hovering_over_quad = ""
        if pos_frames in annotations: # if there are annotations for this frame
            for ai in range(len(annotations[pos_frames])):
                an = annotations[pos_frames][ai]
                if not zoom_rect:
                    ex = [int(an[0]/size_ratio_x),int(an[1]/size_ratio_y),int(an[2]/size_ratio_x),int(an[3]/size_ratio_y)]
                    cv2.rectangle(displayframe,ex,(0,0,255),1)
                else:
                    ex = [int((an[0]-zx0)/size_ratio_x),int((an[1]-zy0)/size_ratio_y),int(an[2]/size_ratio_x),int(an[3]/size_ratio_y)]
                    cv2.rectangle(displayframe,ex,(0,0,255),1)
      
                # see if we are hovering over this annotation
                #        ex[0]  
                #  ex[1]  +-----------------+ 
                #         |                 |
                #         |                 |
                #         +-----------------+ ex[1]+ex[3]
                #         ex[0]+ex[2]
                #
                if mx/size_ratio_x >= ex[0] and mx/size_ratio_x <= ex[0]+ex[2] and my/size_ratio_y >= ex[1] and my/size_ratio_y <= ex[1]+ex[3]:
                    cv2.rectangle(displayframe,ex,(255,255,0),1)
                    hovering_over = ai
                    # determine which diagonal-delimited quadrant the mouse is in
                    if mx/size_ratio_x < ex[0]+ex[2]//2:
                        if my/size_ratio_y < ex[1]+ex[3]//2:
                            cv2.putText(displayframe,"NW",(ex[0],ex[1]),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                            hovering_over_quad = "NW"
                        else:
                            cv2.putText(displayframe,"SW",(ex[0],ex[1]+ex[3]),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                            hovering_over_quad = "SW"
                    else:
                        if my/size_ratio_y < ex[1]+ex[3]//2:
                            cv2.putText(displayframe,"NE",(ex[0]+ex[2],ex[1]),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                            hovering_over_quad = "NE"
                        else:
                            cv2.putText(displayframe,"SE",(ex[0]+ex[2],ex[1]+ex[3]),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                            hovering_over_quad = "SE"
                    if deleting_mode:
                        cv2.putText(displayframe,"deleting",(ex[0],ex[1]-10),cv2.FONT_HERSHEY_DUPLEX,1,(  0,191,255),2)
                    else:
                        cv2.putText(displayframe,f"hov: {pos_frames}:{hovering_over}",(ex[0],ex[1]-10),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                '''
                if mx >= ex0 and mx <= ex1 and my >= ey0 and my <= ey1:
                    flash = (int(time.time()*3)%2) > 0
                    print(f"flash={flash}")
                    bdr = 1 if zoom_rect else 1
                    color = (0,255,255) if deleting_mode and flash else (0,0,255)
                    cv2.rectangle(displayframe,(
                            int(ex0/size_ratio_x),int(ey0/size_ratio_y),
                        ),(
                            int(ex1/size_ratio_x),int(ey1/size_ratio_y),
                        ),color,bdr)
                    rect_transp_fill(displayframe,(
                        int(ex0/size_ratio_x),
                        int(ey0/size_ratio_y),
                        int(ex1/size_ratio_x),
                        int(ey1/size_ratio_y)),color,0.3,bdr)
                    break
                    '''
        cv2.putText(displayframe,f"{findex:04d}:{pos_frames:03d}{deleting_mode}{deleting_ann}",(10,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
        cv2.imshow('frame', displayframe)
        # print(f"frame redrawn: pos_frames={pos_frames}")
        # check for keyboard events
        k = cv2.waitKeyEx(1) # & 0xFF
        save_flag = False
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
            zoom_rect = False
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
                cur = frames[pos_frames][...,1]
                nxt = frames[pos_frames+1][...,1]
                #cur = cv2.cvtColor(frames[pos_frames],cv2.COLOR_BGR2GRAY)
                #nxt = cv2.cvtColor(frames[pos_frames+1],cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(cur,nxt,None,0.5,3,1,3,5,1.2,0) #cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                flowmag = np.sqrt(flow[...,0]**2+flow[...,1]**2)
                an = annotations[pos_frames][hovering_over]
                flow = flow[int(an[1]):int(an[1]+an[3]),int(an[0]):int(an[0]+an[2])]
                fs = flow.shape
                flow=flow[fs[0]//4:fs[0]*3//4,fs[1]//4:fs[1]*3//4]
                # compute mean of flow over rectangle
                flowx = np.quantile(flow[...,0],0.970)
                flowy = np.quantile(flow[...,1],0.970)
                print(f"flowx={flowx}, flowy={flowy}")
                # apply flow to rectangle
                newan = an.copy()
                newan[0] -= flowx * frames.shape[2]*500.0
                newan[1] -= flowy * frames.shape[1]*500.0
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
                    open(labelsfi,"w").write("")
                    for p_frames in annotations:
                        for ai in range(len(annotations[p_frames])):
                            if p_frames == pos_frames and ai == deleting_ann:
                                continue
                            an = annotations[p_frames][ai]
                            yoloroi = list(rect2yolo(an,ww,wh))
                            open(labelsfi,"a").write(f"{p_frames} 0 {yoloroi[0]} {yoloroi[1]} {yoloroi[2]} {yoloroi[3]}\n")
                    # save pos_frames
                    open("posframes.txt","w").write(f"{pos_frames}")
                    deleting_ann = None
                    pause_mode = True

        elif (k == 81 or k == 65361 or k == ord('a')) and hovering_over is not None: # left arrow
            if hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][0] -= 1
                annotations[pos_frames][hovering_over][2] += 1
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][2] -= 1
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][0] -= 1
                annotations[pos_frames][hovering_over][2] += 1
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][2] -= 1
                save_flag = True
        elif (k == 83 or k == 65363 or k == ord('d')) and hovering_over is not None: # right arrow
            if hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][0] += 1
                annotations[pos_frames][hovering_over][2] -= 1
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][2] += 1
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][0] += 1
                annotations[pos_frames][hovering_over][2] -= 1
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][2] += 1
                save_flag = True
        elif (k == 84 or k == 65364 or k == ord('w')) and hovering_over is not None: # down arrow
            if hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][1] += 1
                annotations[pos_frames][hovering_over][3] -= 1
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][1] += 1
                annotations[pos_frames][hovering_over][3] -= 1
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][3] += 1
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][3] += 1
                save_flag = True
        elif (k == 82 or k == 65362 or k == ord('s')) and hovering_over is not None: # up arrow
            if hovering_over_quad == "NW":
                annotations[pos_frames][hovering_over][1] -= 1
                annotations[pos_frames][hovering_over][3] += 1
                save_flag = True
            elif hovering_over_quad == "NE":
                annotations[pos_frames][hovering_over][1] -= 1
                annotations[pos_frames][hovering_over][3] += 1
                save_flag = True
            elif hovering_over_quad == "SW":
                annotations[pos_frames][hovering_over][3] -= 1
                save_flag = True
            elif hovering_over_quad == "SE":
                annotations[pos_frames][hovering_over][3] -= 1
                save_flag = True
            


        if save_flag:
            open(labelsfi,"w").write("")
            for p_frames in annotations:
                for ai in range(len(annotations[p_frames])):
                    an = annotations[p_frames][ai]
                    yoloroi = list(rect2yolo(an,ww,wh))
                    open(labelsfi,"a").write(f"{p_frames} 0 {yoloroi[0]} {yoloroi[1]} {yoloroi[2]} {yoloroi[3]}\n")
            save_flag = False


        mevent = None
        try:
            mevent = mouseq.get_nowait()
        except queue.Empty:
            mevent = None

        if mevent is not None:
            event = mevent['e']
            mx = mevent['x']
            my = mevent['y']
            print(f"mouse event = {event} ({mx},{my})")
            if event==cv2.EVENT_MBUTTONDOWN and mdblclk:
                mdblclk = False
            elif event==cv2.EVENT_MBUTTONDOWN:
                print(f"clicked {mevent['x']},{mevent['y']}")
                if zoom_rect:
                    # already zoomed, make rectangle
                    mx = mx + zx0
                    my = my + zy0
                # establish rectangle
                xbound = frames[0].shape[1]
                ybound = frames[0].shape[0]
                zx0 = int(mx-120)
                zx1 = int(mx+120)
                zy0 = int(my-80)
                zy1 = int(my+80)
                if zx0 < 0:
                    zx1 -= zx0
                    zx0 -= zx0
                if zy0 < 0:
                    zy1 -= zy0
                    zy0 -= zy0
                if zx1 >= xbound:
                    zx0 -= zx1 - xbound-1
                    zx1 -= zx1 - xbound-1
                if zy1 >= ybound:
                    zy0 -= zy1 - ybound-1
                    zy1 -= zy1 - ybound-1

                zoom_rect = True
            elif event == cv2.EVENT_MBUTTONDBLCLK:
                zoom_rect = False
                mdblclk = True
            elif event == cv2.EVENT_MOUSEMOVE:
                # scan existing annotations to see if we are hovering over one
                '''if pos_frames in annotations:
                    for an in annotations[pos_frames]:
                        if zoom_rect:
                            ex0 = an[0]-zx0
                            ex1 = an[0]+an[2]-zx0
                            ey0 = an[1]-zy0
                            ey1 = an[1]+an[3]-zy0
                        else:
                            ex0 = an[0]
                            ex1 = an[0]+an[2]
                            ey0 = an[1]
                            ey1 = an[1]+an[3]
                        if mx >= ex0 and mx <= ex1 and my >= ey0 and my <= ey1:
                            if deleting_mode:
                                cv2.rectangle(displayframe,(ex0,ey0),(ex1,ey1),(0,255,0),1) #(0,255,255),1)
                            else:
                                cv2.rectangle(displayframe,(ex0,ey0),(ex1,ey1),(0,255,0),1) # (0,0,255),1)
                            break
                            rect_transp_fill(displayframe,(ex0,ey0,ex1,ey1),(0,255,0),0.3,1) #(0,0,255),0.3,1)
                            '''
            elif event == cv2.EVENT_LBUTTONDOWN:
                # scan existing annotations to see if we clicked inside one
                if pos_frames in annotations:
                    for ai in range(len(annotations[pos_frames])):
                        an = annotations[pos_frames][ai]
                        if zoom_rect:
                            cv2.rectangle(displayframe,(int(an[0]-zx0),int(an[1]-zy0)),(int(an[0]+an[2]-zx0),int(an[1]+an[3]-zy0)),(0,255,255),1)
                            ex0 = an[0]-zx0
                            ex1 = an[0]+an[2]-zx0
                            ey0 = an[1]-zy0
                            ey1 = an[1]+an[3]-zy0

                        else:
                            # cv2.rectangle(displayframe,(an[0],an[1]),(an[0]+an[2],an[1]+an[3]),(0,0,255),1)
                            cv2.rectangle(displayframe,an,(0,255,255),1)
                            ex0 = an[0]
                            ex1 = an[0]+an[2]
                            ey0 = an[1]
                            ey1 = an[1]+an[3]
                        if mx >= ex0 and mx <= ex1 and my >= ey0 and my <= ey1:
                            #editing_rect = True
                            pass
                            #break

                #cv2.rectangle(displayframe,(1,1),(displayframe.shape[1]-2,displayframe.shape[0]-2),(0,0,255),2)
                roi = list(cv2.selectROI('frame', displayframe, True,False))
                if roi is None:
                    continue
                print(roi)
                if tuple(roi) != (0,0,0,0):
                    roi = [ int(roi[0]*size_ratio_x), int(roi[1]*size_ratio_y), int(roi[2]*size_ratio_x), int(roi[3]*size_ratio_y)]
                    print(roi)
                    if zoom_rect:
                        roi = [roi[0]+zx0, roi[1]+zy0,roi[2],roi[3]]
                    if pos_frames in annotations:
                        annotations[pos_frames].append(roi)
                    else:
                        annotations[pos_frames] = [ roi ]

                    os.makedirs(os.path.dirname(labelsfi), exist_ok=True)
                    ww = frames.shape[2] #displayframe.shape[1]
                    wh = frames.shape[1] #displayframe.shape[0]
                    # convert to yolo format
                    yoloroi = rect2yolo(roi,ww,wh)
                    # write to file - we know it's a new annotation so we can append
                    open(labelsfi,"a").write(f"{pos_frames} 0 {yoloroi[0]} {yoloroi[1]} {yoloroi[2]} {yoloroi[3]}\n");
                    # save pos_frames
                    open("posframes.txt","w").write(f"{pos_frames}")
                else:
                    # if selecting an ROI was cancelled, we are done
                    '''
                    if editing_rect:
                        if pos_frames in annotations:
                            for i in range(len(annotations[pos_frames])):
                                an = annotations[pos_frames][i]
                                an = a
                                if zoom_rect:
                                    ex0 = an[0]-zx0
                                    ex1 = an[0]+an[2]-zx0
                                    ey0 = an[1]-zy0
                                    ey1 = an[1]+an[3]-zy0
                                else:
                                    ex0 = an[0]
                                    ex1 = an[0]+an[2]
                                    ey0 = an[1]
                                    ey1 = an[1]+an[3]
                                if mx >= ex0 and mx <= ex1 and my >= ey0 and my <= ey1:
                                    del annotations[pos_frames][i]
                                    # write to file
                                    # must rewrite the entire file
                                    open(labelsfi,"w").write("")
                                    for frame in annotations:
                                        for an in annotations[pos_frames]:
                                            yoloroi = rect2yolo(an,ww,wh)
                                            open(labelsfi,"a").write(f"{frame} 0 {yoloroi[0]} {yoloroi[1]} {yoloroi[2]} {yoloroi[3]}\n")
                                    break


                    editing_rect = False
                    '''
                cv2.setMouseCallback('frame',mouse_event)

        """
        if mevent is not None:
            if mevent['e'] == cv2.EVENT_MOUSEMOVE:
                if dragging_mode:
                    rx1 = mevent['x']
                    ry1 = mevent['y']
                while True:
                    try:
                        mevent = mouseq.get_nowait()
                    except queue.Empty:
                        break
                    if mevent['e'] == cv2.EVENT_MOUSEMOVE:
                        if dragging_mode:
                            rx1 = mevent['x']
                            ry1 = mevent['y']
                    else:
                        break

        if mevent is not None:
            if mevent['e'] == cv2.EVENT_LBUTTONDOWN:
                # establish rectangle
                rx0 = np.clip(mevent['x']-50,0,frame.shape[1]-1)
                rx1 = np.clip(mevent['x']+50,0,frame.shape[1]-1)
                ry0 = np.clip(mevent['y']-50,0,frame.shape[0]-1)
                ry1 = np.clip(mevent['y']+50,0,frame.shape[0]-1)
                opt_flow(frames,pos_frames, (rx0,ry0),(rx1,ry1))

            elif mevent['e'] == cv2.EVENT_MOUSEMOVE:
                rx1 = mevent['x']
                ry1 = mevent['y']
            elif mevent['e'] == cv2.EVENT_LBUTTONUP:
                dragging_mode = False
            elif mevent['e'] == cv2.EVENT_MBUTTONDOWN:
                rect_mode = False
            elif mevent['e'] == cv2.EVENT_MBUTTONDBLCLK:
                pass
        if (not pause_mode) or one_shot:
            if pos_frames < len(frames)-2:
                pos_frames += 1
            else:
                pos_frames = 0
            frame = frames[pos_frames].copy()
        one_shot = False
        if not dragging_mode:
            time.sleep(0.0)
        """
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

