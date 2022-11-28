def MakeModel(model_name):
    if model_name == "ViTGen":
        from model.ViTGen import Decoder
    if model_name == "TransGen":
        from model.TransGen import Decoder
        return Decoder()