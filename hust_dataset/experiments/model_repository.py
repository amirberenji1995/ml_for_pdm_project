from keras.layers import Input, Dense
from keras.models import Model


def original_model_creator():
    """
    This is a simple shallow ANN model that uses original 11 time domain features
    """

    input = Input(shape=(11,), name="input")
    emb = Dense(
        units=11,
        activation="relu",
        name="HL1",
    )(input)
    emb = Dense(
        units=11,
        activation="relu",
        name="HL2",
    )(emb)
    predicted_values = Dense(units=7, activation="softmax")(emb)

    return Model(inputs=input, outputs=predicted_values)


def with_additional_features_model_creator():
    """
    This is a simple shallow ANN model that uses original 11 time domain features and the 3-dimensional power consumption features
    """

    input = Input(shape=(14,), name="input")
    emb = Dense(
        units=14,
        activation="relu",
        name="HL1",
    )(input)
    emb = Dense(
        units=14,
        activation="relu",
        name="HL2",
    )(emb)
    predicted_values = Dense(units=7, activation="softmax")(emb)

    return Model(inputs=input, outputs=predicted_values)


def ann_power_classifier():
    """
    This is a simple ANN model that uses original 11 time domain features to predict power consumption in a three-dimensional space
    """

    input = Input(shape=(11,), name="input")
    emb = Dense(
        units=11,
        activation="tanh",
        name="HL1",
    )(input)
    emb = Dense(
        units=11,
        activation="tanh",
        name="HL2",
    )(emb)
    predicted_values = Dense(units=3, activation="softmax")(emb)

    return Model(inputs=input, outputs=predicted_values)
