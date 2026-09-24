import numpy as np
from earthmoving.common import CLASSES, write_csv


def classification_metrics(y, pred):
    cm = np.zeros((5, 5), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    support = cm.sum(1)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(5), where=cm.sum(0) != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(5), where=(precision + recall) != 0)
    overall = {'accuracy': float(cm.trace() / max(1, cm.sum())), 'precision': float(precision.mean()),
               'recall': float(recall.mean()), 'macro_f1': float(f1.mean())}
    per_class = [{'class': c, 'precision': precision[i], 'recall': recall[i],
                  'f1': f1[i], 'support': int(support[i])} for i, c in enumerate(CLASSES)]
    return overall, per_class, cm


def save_confusion(path, cm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 6.3), layout='constrained')
    im = ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted action', ylabel='True action', title='Clip classification: XML ground-truth crops')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
