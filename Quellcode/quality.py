import cv2

import numpy as np


def quality(buffer:int = 15, threshold:float = 0.1) -> any:
    r"""Returns the quality metric function for keras's compile.
    Implementation following this paper https://www.researchgate.net/publication/2671242_Empirical_Evaluation_Of_Automatically_Extracted_Road_Axes

    Args:
        buffer: size of buffer, i.e. max euclidean distance between point x and y so that x and y are still considered to be in each other's buffer area.
        threshold: number from which on an input will be considered as activated (1; else 0).

    Returns:
        function taking tensor y_true and y_pred
    """

    diameter = 2 * buffer + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    dilate_i = 1

    def quality(y_true_batch, y_pred_batch):
        y_true_batch_numpy = y_true_batch.numpy()
        y_pred_batch_numpy = y_pred_batch.numpy()
        batch_size = y_true_batch_numpy.shape[0]
        width, height = (y_true_batch_numpy.shape[1], y_pred_batch_numpy.shape[2])

        qual_sum = 0
        for i in range(0, batch_size):
            y_true = y_true_batch_numpy[i][:, :, 0]

            y_pred_binary = np.copy(y_pred_batch_numpy[i])
            y_pred_binary[y_pred_binary < threshold] = 0.
            y_pred_binary[y_pred_binary >= threshold] = 1.
            y_pred_binary = y_pred_binary.reshape((width, height))

            y_true_buffered = cv2.dilate(y_true, kernel, iterations=dilate_i)
            y_pred_binary_buffered = cv2.dilate(y_pred_binary, kernel, iterations=dilate_i)

            tp = np.sum(np.multiply(y_true_buffered, y_pred_binary))
            fp = np.sum(y_pred_binary) - tp
            fn = np.sum(y_true - np.multiply(y_true, y_pred_binary_buffered))

            denom = tp + fp + fn
            qual = tp / denom if denom > 0.0001 or denom < -0.0001 else 1
            qual_sum += qual

        qual_avg = qual_sum / batch_size
        return qual_avg
    return quality