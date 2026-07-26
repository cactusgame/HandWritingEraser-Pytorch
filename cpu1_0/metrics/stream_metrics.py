import numpy as np


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        total = hist.sum()
        acc = np.diag(hist).sum() / total if total else 0.0
        acc_cls = np.divide(
            np.diag(hist), hist.sum(axis=1),
            out=np.full(self.n_classes, np.nan), where=hist.sum(axis=1) != 0)
        acc_cls = np.nanmean(acc_cls)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iu = np.divide(np.diag(hist), union,
                       out=np.full(self.n_classes, np.nan), where=union != 0)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / total if total else np.zeros(self.n_classes)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        if self.n_classes > 1:
            hand_tp = hist[1, 1]
            hand_precision_den = hist[:, 1].sum()
            hand_recall_den = hist[1, :].sum()
            hand_precision = hand_tp / hand_precision_den if hand_precision_den else 0.0
            hand_recall = hand_tp / hand_recall_den if hand_recall_den else 0.0
            pr_sum = hand_precision + hand_recall
            hand_f1 = (2 * hand_precision * hand_recall / pr_sum) if pr_sum else 0.0
        else:
            hand_precision = hand_recall = hand_f1 = 0.0

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Foreground Mean IoU": np.nanmean(iu[1:]) if self.n_classes > 1 else mean_iu,
            "Handwriting IoU": iu[1] if self.n_classes > 1 else mean_iu,
            "Handwriting Precision": hand_precision,
            "Handwriting Recall": hand_recall,
            "Handwriting F1": hand_f1,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
