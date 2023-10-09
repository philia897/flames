import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

from .logger import create_id_by_timestamp

class DBHandlerException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

@dataclass
class DBItem:
    pass

class Items(Enum):
    MODEL_INFO = 1
    CONDITION_CLASS = 2
    EVAL_RESULT = 3

@dataclass
class ModelInfoItem(DBItem):
    # Saved in model_list.json
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    checkpoint_file: str = ""
    model_config_file: str = ""
    meta: Dict = field(default_factory=dict)
    # Saved in condition_classes.json
    # conditions: List = field(default_factory=list)

    def get_dict(self):
        return {
            "parents": self.parents,
            "children": self.children,
            "checkpoint_file": self.checkpoint_file,
            "model_config_file": self.model_config_file,
            "meta": self.meta
        }

@dataclass
class ConditionClassItem(DBItem):
    # Saved in condition_classes.json
    conditions: List = field(default_factory=list)

@dataclass
class EvalResultItem(DBItem):
    # Saved in model_eval_results.json
    eval_results_per_condition: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class DBHandler(object):
    def __init__(self) -> None:
        pass

    def create(self, item:DBItem, item_mode:Items):
        pass

    def read(self, concls_id, item_mode:Items)->DBItem:
        pass

    def update(self, item:DBItem, item_mode:Items, concls_id):
        pass

    def delete(self, concls_id, item_mode:Items):
        pass

class JsonDBHandler(DBHandler):
    def __init__(self, db_path) -> None:
        super().__init__()
        self.db_path = db_path
        self.condition_classes_mapfile = os.path.join(db_path, "condition_classes.json")
        self.model_list_file = os.path.join(db_path, "model_list.json")
        self.eval_results_file = os.path.join(db_path, "model_eval_results.json")
        self.model_cache_path = os.path.join(db_path, "models")
        self._check_db_valid()
        self.concls_id_list = self._get_concls_id_list()
        self.latest_concls_id = max(self.concls_id_list)

    @classmethod
    def create_db(cls, db_path) -> "JsonDBHandler":
        '''Create the DB if it does not exist (won't affect existing DB)'''
        condition_classes_mapfile = os.path.join(db_path, "condition_classes.json")
        model_list_file = os.path.join(db_path, "model_list.json")
        eval_results_file = os.path.join(db_path, "model_eval_results.json")
        model_cache_path = os.path.join(db_path, "models")
        try:
            if not os.path.exists(db_path):
                os.mkdir(db_path)
            if not os.path.exists(condition_classes_mapfile):
                with open(condition_classes_mapfile, "w") as f:
                    json.dump({}, f)
            if not os.path.exists(model_list_file):
                with open(model_list_file, "w") as f:
                    json.dump({}, f)
            if not os.path.exists(eval_results_file):
                with open(eval_results_file, "w") as f:
                    json.dump({}, f)
            if not os.path.exists(model_cache_path):
                os.mkdir(model_cache_path)
            return cls(db_path)
        except Exception as e:
            raise DBHandlerException(f"Failed to create DB: {e}") from e

    def _check_db_valid(self):
        if not os.path.exists(self.condition_classes_mapfile):
            raise DBHandlerException(f"Check DB File System Failed: {self.condition_classes_mapfile} does not exist.")
        elif not os.path.exists(self.model_list_file):
            raise DBHandlerException(f"Check DB File System Failed: {self.model_list_file} does not exist.")
        elif not os.path.exists(self.eval_results_file):
            raise DBHandlerException(f"Check DB File System Failed: {self.eval_results_file} does not exist.")
        elif not os.path.exists(self.model_cache_path):
            raise DBHandlerException(f"Check DB File System Failed: {self.model_cache_path} does not exist.")
        else:
            return True

    def _get_concls_id_list(self):
        with open(self.model_list_file, "r") as f:
            model_list:dict = json.load(f)
            concls_ids = list(model_list.keys())
            return concls_ids

    def _check_concls_id(self, concls_id):
        if not str(concls_id) in self.concls_id_list:
            raise DBHandlerException(f"Concls id {concls_id} does not exist, please check.")

    def _get_new_id(self):
        return str(int(self.latest_concls_id)+1)

    def _record_new_id(self, new_id:str):
        self.concls_id_list.append(new_id)
        self.latest_concls_id = new_id

    def suggest_model_save_path(self, model_name:str=None):
        '''Suggest a model path to save, if model_name is not given, it will use a auto-generated name'''
        if not model_name:
            model_name = f"model-{create_id_by_timestamp()}.pth"
        return os.path.join(self.model_cache_path, model_name)

    def _read_modelinfo(self, concls_id)->ModelInfoItem:
        modelinfo = ModelInfoItem()
        with open(self.model_list_file, "r") as f:
            model_list:dict = json.load(f)
            item = model_list.get(concls_id)
            modelinfo.parents = item['parents']
            modelinfo.children = item['children']
            modelinfo.checkpoint_file = item['checkpoint_file']
            modelinfo.model_config_file = item['model_config_file']
            f.close()
        return modelinfo
    
    def _read_conditionclass(self, concls_id)->ConditionClassItem:
        conditionclass = ConditionClassItem()
        with open(self.condition_classes_mapfile, "r") as f:
            cons:dict = json.load(f)
            item = cons.get(concls_id)
            conditionclass.conditions = item
            f.close()
        return conditionclass
    
    def _read_evalresult(self, concls_id)->EvalResultItem:
        evalreults = EvalResultItem()
        with open(self.eval_results_file, "r") as f:
            results:dict = json.load(f)
            item = results.get(concls_id)
            evalreults.eval_results_per_condition = item
            f.close()
        return evalreults

    def _create_modelinfo(self, modelinfo:ModelInfoItem, new_id:str):
        with open(self.model_list_file, "r+") as f:
            model_list = json.load(f)
            model_list[new_id] = modelinfo.get_dict()
            for parent in modelinfo.parents:  # maintain the graph
                model_list[parent]["children"].append(new_id)
            f.seek(0)  # Move the file cursor to the beginning
            json.dump(model_list, f, indent=4)
            f.truncate()  # Truncate the remaining contents (if any)
            f.close()

    def _create_update_conditionclass(self, conditionclass:ConditionClassItem, concls_id:str):
        with open(self.condition_classes_mapfile, "r+") as f:
            cons = json.load(f)
            cons[concls_id] = conditionclass.conditions
            f.seek(0)
            json.dump(cons, f, indent=4)
            f.truncate()
            f.close()

    def _create_update_evalresult(self, evalresult:EvalResultItem, concls_id:str):
        with open(self.eval_results_file, "r+") as f:
            results = json.load(f)
            results[concls_id] = evalresult.eval_results_per_condition
            f.seek(0)
            json.dump(results, f, indent=4)
            f.truncate()
            f.close()

    def _update_modelinfo(self, modelinfo:ModelInfoItem, concls_id:str):
        with open(self.model_list_file, "r+") as f:
            model_list = json.load(f)    
            try:
                model_list[concls_id]["checkpoint_file"] = modelinfo.checkpoint_file
                model_list[concls_id]["model_config_file"] = modelinfo.model_config_file
                model_list[concls_id]["meta"] = modelinfo.meta
            except Exception as e:
                raise DBHandlerException(f"Failed to update the dedicated model info: {e}") from e
            f.seek(0)  # Move the file cursor to the beginning
            json.dump(model_list, f, indent=4)
            # Truncate the remaining contents (if any)
            f.truncate()

    def read(self, concls_id:str, item_mode:Items)->DBItem:
        '''Get the model item'''
        self._check_concls_id(concls_id)
        if item_mode == Items.MODEL_INFO:
            return self._read_modelinfo(concls_id)
        elif item_mode == Items.CONDITION_CLASS:
            return self._read_conditionclass(concls_id)
        elif item_mode == Items.EVAL_RESULT:
            return self._read_evalresult(concls_id)
        else:
            raise DBHandlerException(f"Invalid Item Mode: {item_mode}")

    def create(self, modelinfo_item: ModelInfoItem, conditionclass_item: ConditionClassItem=None, evalresult_item: EvalResultItem=None)->str:
        '''Add a new model item'''
        new_id = self._get_new_id()
        self._create_modelinfo(modelinfo_item, new_id)
        if not conditionclass_item:
            conditionclass_item = ConditionClassItem([])
        self._create_update_conditionclass(conditionclass_item, new_id)
        if not evalresult_item:
            evalresult_item = EvalResultItem(dict())
        self._create_update_evalresult(evalresult_item, new_id) # Empty eval result item
        self._record_new_id(new_id)
        return new_id

    def update(self, item:DBItem, item_mode:Items, concls_id:str):
        '''Update the item and return its concls_id'''
        self._check_concls_id(concls_id)
        if item_mode == Items.MODEL_INFO:
            self._update_modelinfo(item, concls_id)
        elif item_mode == Items.CONDITION_CLASS:
            self._create_update_conditionclass(item, concls_id)
        elif item_mode == Items.EVAL_RESULT:
            self._create_update_evalresult(item, concls_id)
        else:
            raise DBHandlerException(f"Invalid Item Mode: {item_mode}")
        return concls_id

    def sweep_cache(self, delete_old=False):
        '''
        Find the models saved in the cache but not tracked by model_list
        And delete them if delete_old=True, default False
        '''
        names = os.listdir(self.model_cache_path)
        filenames = [name for name in names if os.path.isfile(os.path.join(self.model_cache_path, name)) and name.endswith(".pth")]
        with open(self.model_list_file, "r") as f:
            model_list:dict = json.load(f)
            for k,item in model_list.items():
                filenames.pop(filenames.index(os.path.basename(item["checkpoint_file"])))
        if delete_old:
            for file in filenames:
                os.remove(os.path.join(self.model_cache_path, file))
        return filenames
        