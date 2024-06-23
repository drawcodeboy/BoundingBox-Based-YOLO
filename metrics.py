from models.yololoss import intersection_over_union
import numpy as np

# AP, mAP, 11점 보간법
# Reference: https://herbwood.tistory.com/3

def AP(detections, groundtruths, classes, IOUThreshold = 0.3, method = 'AP'):
    
    # 클래스별 AP, Precision, Recall 등 관련 정보를 저장할 리스트
    result = []
    
    # 클래스별로 접근
    for c in classes:
		
        # 특정 class에 해당하는 box를 box타입(detected, ground truth)에 따라 분류
        dects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]
		
        # 전체 ground truth box의 수
        # Recall 값의 분모
        npos = len(gts)
		
        # confidence score에 따라 내림차순 정렬
        dects = sorted(dects, key = lambda conf : conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
		
        # 각 이미지별 ground truth box의 수
        # {99 : 2, 380 : 4, ....}
        det = Counter(cc[0] for cc in gts)
        
        # {99 : [0, 0], 380 : [0, 0, 0, 0], ...}
        for key, val in det.items():
            det[key] = np.zeros(val)
        # 전체 detected box
        for d in range(len(dects)):

            # ground truth box 중에서 detected box와 같은 이미지 파일에 존재하는 box
            # dects[d][0] : 이미지 파일명
            gt = [gt for gt in gts if gt[0] == dects[d][0]]

            iouMax = 0
			
            # 하나의 detected box에 대하여 같은 이미지에 존재하는 모든 ground truth 값을 비교
            # 가장 큰 IoU 값을 가지는 하나의 ground truth box에 매칭
            for j in range(len(gt)):
                iou1 = iou(dects[d][3], gt[j][3])
                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j
			
            # IoU 임계값 이상 and ground truth box가 매칭되지 않음 => TP
            # IoU 임계값 미만 or ground truth box가 다른 detected box에 이미 매칭됨 => FP
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
		
        # Precision, Recall 값을 구하기 위한 누적 TP, FP
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        r = {
            'class' : c,
            'precision' : prec,
            'recall' : rec,
            'AP' : ap,
            'interpolated precision' : mpre,
            'interpolated recall' : mrec,
            'total positives' : npos,
            'total TP' : np.sum(TP),
            'total FP' : np.sum(FP)
        }

        result.append(r)

    return result

def ElevenPointInterpolatedAP(rec, prec):
    mrec = [e for e in rec]
    mpre = [e for e in prec]

    # recallValues = [1.0, 0.9, ..., 0.0]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []


    for r in recallValues:
        # r : recall값의 구간
        # argGreaterRecalls : r보다 큰 값의 index
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        print(r, argGreaterRecalls)

        # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11
    
    return [ap, rhoInterp, recallValues, None]

def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']
    mAP = ap / len(result)
    
    return mAP

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    