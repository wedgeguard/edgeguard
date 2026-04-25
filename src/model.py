import tensorflow as tf
from tensorflow.keras import layers, Model

class EdgeGuardFinal:
    """
    EdgeGuard: A Hybrid Simulation and Real-Data–Driven Framework.
    Optimized for NVIDIA Jetson AGX Orin & IEEE/DTISD Publication.
    """
    def __init__(self, mc_samples=50):
        self.mc_samples = mc_samples
        self.model = self._build_model()

    def _build_model(self):
        # --- 1. Inputs ---
        radar_in = layers.Input(shape=(100, 10), name="radar")
        ais_in   = layers.Input(shape=(50, 8), name="ais")
        eoir_in  = layers.Input(shape=(224, 224, 3), name="eoir")

        # --- 2. Vision: MobileNetV2 (Maritime Fine-Tuning) ---
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
        base.trainable = True
        for layer in base.layers[:-20]: layer.trainable = False # Partial Fine-Tuning
        
        x_v = layers.GlobalAveragePooling2D()(base(eoir_in))
        x_v = layers.Dense(32, activation="relu")(x_v)

        # --- 3. Sequential: cuDNN Optimized LSTMs (No recurrent_dropout) ---
        def lstm_block(inp):
            x = layers.LSTM(64, return_sequences=True)(inp)
            x = layers.Dropout(0.2)(x)
            return layers.LSTM(32)(x)

        x_a = lstm_block(ais_in)
        x_r = lstm_block(radar_in)

        # --- 4. Modality Attention (Equation 1) ---
        def get_score(feat, n):
            return layers.Dense(1, name=f"score_{n}")(layers.Dense(32, activation="tanh")(feat))

        scores = layers.Softmax(name="modality_attention")(
            layers.Concatenate()([get_score(x_r, "r"), get_score(x_a, "a"), get_score(x_v, "v")])
        )

        # --- 5. Fusion & Classification ---
        f_r, f_a, f_v = [layers.Dense(32)(f) for f in [x_r, x_a, x_v]]
        
        fused = layers.Add()([
            layers.Multiply()([f_r, layers.Lambda(lambda x: x[:, 0:1])(scores)]),
            layers.Multiply()([f_a, layers.Lambda(lambda x: x[:, 1:2])(scores)]),
            layers.Multiply()([f_v, layers.Lambda(lambda x: x[:, 2:3])(scores)])
        ])

        x = layers.Dense(64, activation="relu")(fused)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(3, activation="softmax")(x)
        
        return Model(inputs=[radar_in, ais_in, eoir_in], outputs=out)

    @tf.function
    def predict_with_uncertainty(self, radar, ais, eoir, mc_samples=None):
        """Vectorized MC Dropout Inference (Equation 3)"""
        T = mc_samples if mc_samples is not None else self.mc_samples
        
        r_t = tf.tile(radar, [T, 1, 1])
        a_t = tf.tile(ais,   [T, 1, 1])
        v_t = tf.tile(eoir,  [T, 1, 1, 1])
        
        preds = self.model([r_t, a_t, v_t], training=True)
        preds = tf.reshape(preds, (T, -1, 3))
        
        mean = tf.reduce_mean(preds, axis=0) # ŷ
        var  = tf.math.reduce_variance(preds, axis=0) # σ²
        std  = tf.math.sqrt(var) # σ
        
        return mean, var, std
