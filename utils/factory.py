from models.foster_uda import FOSTER_UDA
def get_model(model_name, args):
    name = model_name.lower()
    if name == "foster_uda":
        return FOSTER_UDA(args)
    else:
        assert 0, "Not Implemented!"
