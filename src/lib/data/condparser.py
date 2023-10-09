from enum import Enum
from typing import List, Dict

class ConditionParserMode(Enum):
    '''
    Support totally three kinds of condition format:
    String, List and Dict.
    Normally String as a key for database storing
    List as parameters for processing
    Dict show details for debugging and visualization
    '''
    STRING_KEY = 1
    VALUE_LIST = 2
    DETAIL_DICT = 3

class ConditionParser:
    def convert(condition, target_mode):
        pass

class BDD100KConditionParser(ConditionParser):
    attributes = ["weather", "scene", "timeofday"]
    
    @classmethod
    def _attr_string_to_list(cls, s:str):
        return s.split('-')
    
    @classmethod
    def _attr_list_to_string(cls, l:List):
        return '-'.join(l)

    @classmethod
    def _attr_list_to_dict(cls, l:List):
        return dict(zip(cls.attributes, l))

    @classmethod
    def _attr_dict_to_list(cls, d:Dict):
        return [d[k] for k in cls.attributes]

    @classmethod
    def convert(cls, condition, target_mode, source_mode=None):
        '''
        convert the condition to specific format, 
        source_mode param only for hint, no real use.
        '''
        if isinstance(condition, str):
            condition = cls._attr_string_to_list(condition)
        elif isinstance(condition, dict):
            condition = cls._attr_dict_to_list(condition)
        elif isinstance(condition, list):
            pass
        else:
            raise ValueError(f"Invalid condition type {type(condition)}")
        if target_mode == ConditionParserMode.STRING_KEY:
            return cls._attr_list_to_string(condition)
        elif target_mode == ConditionParserMode.DETAIL_DICT:
            return cls._attr_list_to_dict(condition)
        elif target_mode == ConditionParserMode.VALUE_LIST:
            return condition
        else:
            raise ValueError(f"Invalid target mode: {target_mode}")