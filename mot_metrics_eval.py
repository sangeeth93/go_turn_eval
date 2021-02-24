import torch,glob,cv2,json,sys,pdb,pickle,subprocess, pdb, os
import numpy as np
import torch.nn as nn
import model
import motmetrics as mm
from tqdm import tqdm


def load_data(path):
    #Load data from the json file with the text locations and track annotations.
    #json file to be in the format of Scalabel tool output.
    with open(path,"r") as f:
        gt_raw = json.load(f)
    gt={}
    for item in gt_raw:
      if item["videoName"] not in gt.keys():
        gt[item["videoName"]]={}
      if item["index"] not in gt[item["videoName"]].keys():
        gt[item["videoName"]][item["index"]]=[]
      if item["labels"] != None:
        gt[item["videoName"]][item["index"]].append(item["labels"])
    return gt

def load_vid_frames(path):
    op={}
    if os.path.isfile(path):
        vidcap=cv2.VideoCapture(path)
        f=-1
        while (vidcap.isOpened()):
            success,image=vidcap.read()
            if success==True:
                f+=1
                op[f]=image
            else:
                break
    else:
        print(path," doesnt exist")
        sys.exit()
    return op


def check_labels_in_frame(vid,f,s,gt):
    #check if the same text instance is present in the following frames of distance upto s.
    #return a list of items with all the corresponding text locations in the frames.
    # vid=int(vid)
    tmp_lst=[]
    if f in gt[vid].keys() and s in gt[vid].keys():
        # pdb.set_trace()
        if gt[vid][s]:
            lbls2 = gt[vid][s][0]
        else:
            lbls2 = gt[vid][s]
        if gt[vid][f]:
            lbls1 = gt[vid][f][0]
        else:
            lbls1 = gt[vid][f]
        for lbl1 in lbls1:
          for lbl2 in lbls2:
            # print(lbl1,lbl1[0])
            # print(lbl2,lbl2[0])
            # sys.exit()
            # print(lbl1["id"],lbl2[0]["id"])
            # if (lbl1[0]["id"] == lbl2[0]["id"]) and ((lbl1[0]["category"] == "European" and lbl2[0]["category"] == "European") or (lbl1[0]["category"] == "English" and lbl2[0]["category"] == "English") or (lbl1[0]["category"] == "Asian" and lbl2[0]["category"] == "Asian")):
            if (lbl1["id"] == lbl2["id"]) and ((lbl1["category"] == "English_Legible" and lbl2["category"] == "English_Legible") or (lbl1["category"] == "Non_English_Legible" and lbl2["category"] == "Non_English_Legible")):
                b1 = [round(lbl1["box2d"]["x1"]),round(lbl1["box2d"]["y1"]),round(lbl1["box2d"]["x2"]),round(lbl1["box2d"]["y2"])]
                b2 = [round(lbl2["box2d"]["x1"]),round(lbl2["box2d"]["y1"]),round(lbl2["box2d"]["x2"]),round(lbl2["box2d"]["y2"])]
                tmp_lst.append([f,s,b1,b2,lbl1["id"]])
    # print(tmp_lst)
    return tmp_lst


def track_text(f1, f2, loc_tar, loc_srch, mdl, dev):

    image = f1
    image_srch = f2

    tar = np.zeros((320,240,3))
    srch = np.zeros((320,240,3))

    x_srch_len = (loc_srch[2]-loc_srch[0])
    y_srch_len = (loc_srch[3]-loc_srch[1])
    x_srch_c = loc_srch[0] + int(x_srch_len/2)
    y_srch_c = loc_srch[1] + int(y_srch_len/2)

    x_len = (loc_tar[2]-loc_tar[0])
    y_len = (loc_tar[3]-loc_tar[1])
    x_c = loc_tar[0] + int(x_len/2)
    y_c = loc_tar[1] + int(y_len/2)

    x_append = 120 - int(x_len/2)
    y_append = 160 - int(y_len/2)

    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if x_len <= 160 and y_len <= 120:
        # print("smaller than 160 * 120")
        if y_c-int(y_len/2)-y_append < 0:
            left_pad = y_append+int(y_len/2)-y_c
        if y_c+int(y_len/2)+y_append > image.shape[0]:
            right_pad = y_c+int(y_len/2)+y_append - image.shape[0]
        if x_c-int(x_len/2)-x_append < 0:
            top_pad = x_append+int(x_len/2)-x_c
        if x_c+int(x_len/2)+x_append > image.shape[1]:
            bot_pad = x_c+int(x_len/2)+x_append - image.shape[1]
    mean_val = np.mean(image,axis=(0,1))
    mean_val_srch = np.mean(image_srch,axis=(0,1))

    x1 = x_c-int(x_len/2)-x_append
    y1 = y_c-int(y_len/2)-y_append
    x2 = x_c+int(x_len/2)+x_append
    y2 = y_c+int(y_len/2)+y_append

    if left_pad or right_pad or top_pad or bot_pad:
        # print("Padding done")
        if left_pad:
            y1 = 0
        if right_pad:
            y2 = 720
        if top_pad:
            x1 = 0
        if bot_pad:
            x2 = 1280

        tar_temp = image[y1:y2,x1:x2]

        tar[left_pad:320-right_pad,top_pad:240-bot_pad] = tar_temp

        tar[:,:,0][tar[:,:,0]==0] = mean_val[0]
        tar[:,:,1][tar[:,:,1]==0] = mean_val[1]
        tar[:,:,2][tar[:,:,2]==0] = mean_val[2]
        
        srch_temp = image_srch[y1:y2,x1:x2]

        srch[left_pad:320-right_pad,top_pad:240-bot_pad] = srch_temp

        srch[:,:,0][srch[:,:,0]==0] = mean_val_srch[0]
        srch[:,:,1][srch[:,:,1]==0] = mean_val_srch[1]
        srch[:,:,2][srch[:,:,2]==0] = mean_val_srch[2]

    else:
        # print("Padding not done")
        x1 = x_c-int(x_len/2)-x_append
        y1 = y_c-int(y_len/2)-y_append
        x2 = x_c+int(x_len/2)+x_append
        y2 = y_c+int(y_len/2)+y_append
        tar = image[y1:y2,x1:x2]
        srch = image_srch[y1:y2,x1:x2]


    srch_xc = 160 + x_c - x_srch_c
    srch_yc = 120 + y_c - y_srch_c
    x_len = int(x_srch_len/2)
    y_len = int(y_srch_len/2)

    tar_img = torch.tensor(tar).permute(2,0,1).float().unsqueeze(0).to(dev)
    srch_img = torch.tensor(srch).permute(2,0,1).float().unsqueeze(0).to(dev)

    # pdb.set_trace()
    try:
        pred = mdl(tar_img, srch_img)
        pred = pred.squeeze(0).detach().cpu().numpy()
        pd = [int(pred[0]*320),int(pred[1]*240),int(pred[2]*320),int(pred[3]*240)]
        # cv2.imwrite("srch_b4.jpg", srch)
        # cv2.imwrite("full_srch_b4.jpg", image_srch)
        # pdb.set_trace()
        x1_sc = pd[0]-pd[2]
        y1_sc = pd[1]-pd[3]
        x2_sc = pd[0]+pd[2]
        y2_sc = pd[1]+pd[3]
    except:
        # print("Error")
        pd = [0,0,0,0]
        x1 = 0
        y1 = 0
        # pdb.set_trace()
    

    # cv2.rectangle(srch,(x1_sc,y1_sc),(x2_sc,y2_sc),(255,0,0),3)
    # cv2.rectangle(srch,(srch_xc-x_len,srch_yc-y_len),(srch_xc+x_len,srch_yc+y_len),(0,255,0),3)

    # cv2.rectangle(image_srch,(x1+x1_sc,y1+y1_sc),(x1+x2_sc,y1+y2_sc),(255,0,0),3)
    # cv2.rectangle(image_srch,(loc_srch[0],loc_srch[1]),(loc_srch[2],loc_srch[3]),(0,255,0),3)

    # cv2.imwrite("srch.jpg", srch)
    # cv2.imwrite("full_srch.jpg", image_srch)

    return pd,x1,y1
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = nn.DataParallel(model.GoNet())
if os.path.exists("go_turn.pth"):
    m.load_state_dict(torch.load("go_turn.pth")['state_dict'])
    print("pretrained model loaded")
else:
    print("Couldnot find trained model. Exitting")
    sys.exit()
m.to(device)
m.eval()
op_summary =""
acc = mm.MOTAccumulator(auto_id=True)
folders = [700,800,900]
for fldr in folders:
    if not os.path.exists("/ssd_scratch/cvit/bsr/"):
        op = subprocess.run(["mkdir","/ssd_scratch/cvit/bsr"])
    if not os.path.exists("/ssd_scratch/cvit/bsr/"+fldr):
        op = subprocess.run(["scp","-r","bsr@ada:/share3/bsr/"+fldr,"/ssd_scratch/cvit/bsr/"])

    vids = glob.glob("/ssd_scratch/cvit/bsr/"+fldr+"/*.mp4")
    # video_location = "/home/bsr/goturn/eval_vid/"+vid+".mp4"
    annotation_file_path = "/ssd_scratch/cvit/bsr/"+fldr+"/"+fldr+"_videos_results.json"
    gt = load_data(annotation_file_path)
    for video_location in tqdm(vids):
        vid = video_location.split("/")[-1].replace(".mp4","")
        frames_vid = load_vid_frames(video_location)

        frames = sorted(frames_vid)
        op_frames={}
        for f_no in frames:
            op_frames[f_no+1]=[]
            t_lst = check_labels_in_frame(vid,f_no,f_no+1,gt)
            prev_pred={}
            if t_lst:
                # if len(t_lst)>1:
                #     print("more than one object found")
                #     pdb.set_Trace()
                for item in t_lst:
                    if item[4] not in prev_pred.keys():
                        pd,x1,y1 = track_text(frames_vid[item[0]],frames_vid[item[1]],item[2],item[3],m, device) #(x1+pd[0]-pd[2],y1+pd[1]-pd[3]),(x1+pd[0]+pd[2],y1+pd[1]+pd[3])
                        # cv2.rectangle(frames_vid[item[1]],(x1+pd[0]-pd[2],y1+pd[1]-pd[3]),(x1+pd[0]+pd[2],y1+pd[1]+pd[3]),(255,0,0),3)
                        # cv2.rectangle(frames_vid[item[1]],(item[3][0],item[3][1]),(item[3][2],item[3][3]),(0,255,0),3)
                        # cv2.imwrite("srch.jpg",frames_vid[item[1]])
                        # sys.exit()
                        prev_pred[item[4]] = [x1+pd[0]-pd[2],y1+pd[1]-pd[3],x1+pd[0]+pd[2],y1+pd[1]+pd[3]]
                    else:
                        pd,x1,y1 = track_text(frames_vid[item[0]],frames_vid[item[1]],prev_pred[item[4]][0],item[3],m, device)

                        #commented out code that gives ground truth once the previous prediction is nothing
                        # if prev_pred[item[4]] == [0,0,0,0]:
                        #     pd,x1,y1 = track_text(frames_vid[item[0]],frames_vid[item[1]],item[2],item[3],m, device)
                        # else:
                        #     pd,x1,y1 = track_text(frames_vid[item[0]],frames_vid[item[1]],prev_pred[item[4]][0],item[3],m, device)
            
                        prev_pred[item[4]] = [x1+pd[0]-pd[2],y1+pd[1]-pd[3],x1+pd[0]+pd[2],y1+pd[1]+pd[3]]
                    # if op_frames[f_no+1]==[]:
                    #     op_frames[f_no+1] = []

                    op_frames[f_no+1].append([prev_pred[item[4]],item[3]])

                    # op_frames.append(frames_vid[item[1]],prev_pred[item[4]],item[3])
        # size = (frames_vid[0].shape[1], frames_vid[0].shape[0])
        # result = cv2.VideoWriter('filename.avi',  
                                 # cv2.VideoWriter_fourcc(*'MJPG'), 
                                 # 30, size) 


        for k in op_frames.keys():
            pred_boxes = []
            gt_boxes = []
            try:
                img = frames_vid[k]
                for item in op_frames[k]:
                    x_width = int((item[0][2]-item[0][0])/2)
                    x_c = int(item[0][0]+x_width)
                    y_width = int((item[0][3]-item[0][1])/2)
                    y_c = int(item[0][1]+y_width)
                    pred_boxes.append([x_c,y_c,x_width,y_width])
                    x_width = (item[1][2]-item[1][0])
                    x_c = int(item[1][0]+x_width)
                    y_width = (item[1][3]-item[1][1])
                    y_c = int(item[1][1]+y_width)
                    gt_boxes.append([x_c,y_c,x_width,y_width])
                    # pred_boxes = pred_boxes
                    # gt_boxes = gt_boxes
                a = range(1,len(op_frames[k])+1)
                b = range(1,len(op_frames[k])+1)
                if pred_boxes:
                    dist = mm.distances.iou_matrix(pred_boxes, gt_boxes, max_iou=0.5)
                    # pdb.set_trace()
                    acc.update(a,b,dist)
            except:
                pass
                # result.release()
                # sys.exit()

mh = mm.metrics.create


summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=mm.metrics.motchallenge_metrics,
    names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
with open("mot_summary.txt","w+") as g:
    g.write(strsummary)
print(strsummary)
# summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
#     # pdb.set_trace()
# print(summary)

        # pdb.set_trace()


#         cv2.rectangle(img,(pred_box[0],pred_box[1]),(pred_box[2],pred_box[3]),(255,0,0),3)
#         cv2.rectangle(img,(gt_box[0],gt_box[1]),(gt_box[2],gt_box[3]),(0,255,0),3)
#     result.write(img)
# result.release()





