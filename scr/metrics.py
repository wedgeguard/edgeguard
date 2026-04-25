import tensorflow as tf

def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (Equation 2 in paper).
    Measures the gap between predictive confidence and empirical accuracy.
    """
    bin_boundaries = tf.linspace(0., 1., n_bins + 1)
    confidences = tf.reduce_max(probs, axis=1)
    predictions = tf.argmax(probs, axis=1)
    accuracies = tf.cast(tf.equal(predictions, tf.cast(labels, tf.int64)), tf.float32)
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = tf.logical_and(confidences > bin_boundaries[i], 
                                confidences <= bin_boundaries[i+1])
        prop_in_bin = tf.reduce_mean(tf.cast(in_bin, tf.float32))
        if prop_in_bin > 0:
            acc_in_bin = tf.reduce_mean(accuracies[in_bin])
            conf_in_bin = tf.reduce_mean(confidences[in_bin])
            ece += prop_in_bin * tf.abs(acc_in_bin - conf_in_bin)
    return ece
