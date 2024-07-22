from othermodels.ALAE import ALAE_TAE





def get_model_by_name(name, ts_len, num_sensors, num_classes):

    model_dict = {
        "ALAE_TAE": ALAE_TAE((1, 1, ts_len, num_sensors), num_classes),
    }
    return model_dict[name]
