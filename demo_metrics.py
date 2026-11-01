import numpy as np, json, pathlib
np.random.seed(0)
def dice(pred,true):
    pred=pred.astype(bool); true=true.astype(bool); inter=(pred&true).sum(); return float(2*inter/(pred.sum()+true.sum()+1e-9))
def iou(pred,true):
    pred=pred.astype(bool); true=true.astype(bool); inter=(pred&true).sum(); union=(pred|true).sum(); return float(inter/(union+1e-9))
# SYNTHETIC masks demo
true=np.zeros((128,128),dtype=int); true[30:90,30:90]=1; true[50:70,50:70]=2
pred=true.copy(); pred[0:12,0:12]=1; pred[30:90,85:90]=0; pred[100:110,100:110]=2
ious=[]; dices=[]
for c in [1,2]:
    ious.append(iou(pred==c,true==c)); dices.append(dice(pred==c,true==c))
summary={"note":"SYNTHETIC masks demo only - NOT dissertation results","demo_mIoU":round(float(np.mean(ious)),4),"demo_mean_Dice":round(float(np.mean(dices)),4),"per_class_IoU":[round(float(x),4) for x in ious],"per_class_Dice":[round(float(x),4) for x in dices]}
pathlib.Path(__file__).parent.joinpath("results").mkdir(exist_ok=True)
pathlib.Path(__file__).parent.joinpath("results/demo_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
