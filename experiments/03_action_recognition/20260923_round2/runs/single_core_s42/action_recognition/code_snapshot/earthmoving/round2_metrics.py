"""Temporal evaluation on explicitly masked truth, with boundary uncertainty intervals."""
import numpy as np
from earthmoving.common import spans
from earthmoving.metrics import classification_metrics


def runs(y):
    if len(y) == 0:
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(y) != 0)+1, len(y)]
    return [(int(y[a]), int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def edit_similarity(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return 1-previous[-1]/max(1, len(a), len(b))


def segment_counts(truth, predicted, threshold):
    gt, pr = runs(truth), runs(predicted)
    matched, tp = set(), 0
    for c, a, b in pr:
        candidates = []
        for i, (d, s, e) in enumerate(gt):
            if c == d:
                intersection = max(0, min(b,e)-max(a,s))
                candidates.append((intersection/(b-a+e-s-intersection), i))
        if candidates:
            overlap, i = max(candidates, key=lambda x: (x[0], -x[1]))
            if overlap >= threshold and i not in matched:
                matched.add(i); tp += 1
    return tp, len(pr)-tp, len(gt)-tp


def boundary_stats(truth, predicted, tolerance=16):
    """Pair-aware matching. Ground-truth transition location is an interval, never guessed."""
    gt, pr = [], []
    for a, b in spans(predicted >= 0):
        known = np.flatnonzero(truth[a:b] >= 0)+a
        for left, right in zip(known[:-1], known[1:]):
            if truth[left] != truth[right]:
                gt.append((int(truth[left]), int(truth[right]), int(left+1), int(right)))
        for t in range(a+1,b):
            if predicted[t-1] != predicted[t] and truth[t-1] >= 0 and truth[t] >= 0:
                pr.append((int(predicted[t-1]), int(predicted[t]), t))
    # Predictions inside the uncertain truth gaps are intentionally not scored as exact boundaries.
    # Add them only where their surrounding known truth brackets define a scored transition.
    for a,b,lo,hi in gt:
        for t in range(lo,hi+1):
            if t > 0 and predicted[t-1] == a and predicted[t] == b and (a,b,t) not in pr:
                pr.append((a,b,t))
    used, errors = set(), []
    for a,b,t in sorted(pr, key=lambda x:x[2]):
        candidates = [(max(lo-t, t-hi, 0),i) for i,(c,d,lo,hi) in enumerate(gt) if a==c and b==d and i not in used]
        if candidates:
            distance,i = min(candidates)
            if distance <= tolerance:
                used.add(i); errors.append(distance)
    return {'tolerance_frames': tolerance, 'matched': len(errors), 'true_boundaries':len(gt),
            'predicted_scored_boundaries':len(pr), 'precision':len(errors)/max(1,len(pr)),
            'recall':len(errors)/max(1,len(gt)),
            'matched_distance_to_truth_bracket_mean_frames':float(np.mean(errors)) if errors else None,
            'boundary_rule':'class-pair-aware nearest one-to-one matching; unknown-only transitions unscored'}


def temporal_metrics(truth, predicted):
    mask = (truth >= 0) & (predicted >= 0)
    metrics, classes, cm = classification_metrics(truth[mask],predicted[mask])
    counts = {str(t):np.zeros(3,dtype=int) for t in [.1,.25,.5]}
    edits = []
    for a,b in spans(predicted >= 0):
        valid = truth[a:b] >= 0
        y,p = truth[a:b][valid],predicted[a:b][valid]
        if len(y):
            edits.append(edit_similarity([r[0] for r in runs(y)],[r[0] for r in runs(p)]))
            for t in counts:
                counts[t] += segment_counts(y,p,float(t))
    return {'frame_metrics':metrics, 'class_metrics':classes, 'confusion':cm.tolist(),
            'valid_truth_frames':int((truth>=0).sum()), 'evaluated_frames':int(mask.sum()),
            'valid_truth_without_prediction':int(((truth>=0)&(predicted<0)).sum()),
            'ignored_truth_frames':int((truth<0).sum()), 'coverage':float(mask.sum()/max(1,(truth>=0).sum())),
            'edit':float(np.mean(edits)) if edits else None,
            'segment_counts':{t:v.tolist() for t,v in counts.items()},
            'segment_f1':{t:float(2*v[0]/max(1,2*v[0]+v[1]+v[2])) for t,v in counts.items()},
            'boundary':boundary_stats(truth,predicted),
            'temporal_metric_axis':'remove unknown truth frames inside each prediction-covered component; IoU measured in remaining known-frame counts; do not bridge prediction gaps',
            'exact_boundary_ground_truth_claimed':False}
