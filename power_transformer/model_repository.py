from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def original_regressor_model():
    """
    A shallow ANN to regress the health index and remaining useful life of the power transformer, based on the original 14 features.
    """
    input = Input(shape=(14,), name="input")
    emb = Dense(
        units=9,
        activation="tanh",
        name="HL1",
    )(input)
    emb = Dense(
        units=4,
        activation="tanh",
        name="HL2",
    )(emb)
    predicted_values = Dense(units=2, activation="tanh", name="values")(emb)

    return Model(inputs=input, outputs=predicted_values)


def with_features_regressor_model():
    """
    A shallow ANN to regress the health index and remaining useful life of the power transformer, based on the original 14 features and the additional 7 features.
    """
    input = Input(shape=(21,), name="input")
    emb = Dense(
        units=16,
        activation="tanh",
        name="HL1",
    )(input)
    emb = Dense(
        units=11,
        activation="tanh",
        name="HL2",
    )(emb)
    predicted_values = Dense(units=2, activation="tanh", name="values")(emb)

    return Model(inputs=input, outputs=predicted_values)


def ann_state_classifier():
    input = Input(shape=(14,), name="input")
    emb = Dense(
        units=10,
        activation="relu",
        name="HL1",
    )(input)
    predicted_values = Dense(units=7, activation="softmax")(emb)

    return Model(inputs=input, outputs=predicted_values)
