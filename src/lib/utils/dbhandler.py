import json

class DBHandler(object):
    def __init__(self) -> None:
        pass

    def get_conditions_from_class():
        pass

    def get_model_configs():
        pass

    def save2model_list():
        pass


class JsonDBHandler(DBHandler):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_conditions_from_class(class_file, class_id):
        with open(class_file, "r") as f:
            configs = json.load(f)
        return configs[class_id]

    @staticmethod
    def get_model_configs(model_list_file, class_id):
        with open(model_list_file, "r") as f:
            model_list = json.load(f)
        if class_id in model_list.keys():
            return model_list[class_id]["checkpoint_file"], model_list[class_id]["model_config_file"]
        else:
            raise ValueError(f"Condition Class Id {class_id} does not exist, please check!")

    @staticmethod
    def save2model_list(model_list_file, checkpoint_file, model_config_file, class_id):
        with open(model_list_file, "r+") as f:
            model_list = json.load(f)
            if class_id in model_list.keys():   # Update the dedicated model item
                model_list[class_id]["checkpoint_file"] = checkpoint_file
                model_list[class_id]["model_config_file"] = model_config_file
            else:              # create a new dedicated model item as root
                model_list[class_id] = {
                    "parents": [class_id],
                    "children": [],
                    "checkpoint_file": checkpoint_file,
                    "model_config_file": model_config_file
                }

            f.seek(0)  # Move the file cursor to the beginning
            json.dump(model_list, f, indent=4)
            # Truncate the remaining contents (if any)
            f.truncate()