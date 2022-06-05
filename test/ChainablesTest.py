import sys
sys.path.append('../')

from src.chainables import Chainable, ChainableObject, ChainableFunction, TypeSafeChainableFunction
from typing import List, Optional, Any, Union
from pprint import pprint


class GetRaw(ChainableFunction):
    def __init__(self):
        super().__init__(name="get_raw", uuid="0001")
        super().set_default_params({'debug': True, 'data_name': "raw"})

    def func(self, param):
        super().store('raw', [1, 2, 3, 4],
                      {'type': "list", 'dim': 1, 'name': param['data_name'], 'label': {'de': "Spannung"},
                       'quant': "qudt:Voltage", 'unit': "qudt:mV", 'test': {'nested': "value"}})


class GetRawTypeSafe(TypeSafeChainableFunction):
    class Param(TypeSafeChainableFunction.Param):
        data_name: str = "raw"

    def __init__(self):
        super(__class__, self).__init__(name="get_raw", uuid="0001", param_class=__class__.Param)

    def func(self, param):
        # param.debug
        super().store('raw', [1, 2, 3, 4],
                      {'type': "list", 'dim': 1, 'name': param.data_name, 'label': {'de': "Spannung"},
                       'quant': "qudt:Voltage", 'unit': "qudt:mV", 'test': {'nested': "value"}})


class GetRawFullTypeSafe(TypeSafeChainableFunction):
    class Param(TypeSafeChainableFunction.Param):
        data_name: str = "raw"

    class Data(TypeSafeChainableFunction.Data):
        content: List[int]

    class Meta(TypeSafeChainableFunction.Meta):
        name: str
        label: dict[str, str]
        quant: str

    def __init__(self):
        super(__class__, self).__init__(name="get_raw", uuid="0001", param_class=__class__.Param,
                                        data_class=__class__.Data, meta_class=__class__.Meta)

    def func(self, param):
        super().store('raw', {'content': [1, 2, 3, 4]},
                      {'type': "list", 'dim': 1, 'name': param.data_name, 'label': {'de': "Spannung"},
                       'quant': "qudt:Voltage", 'unit': "qudt:mV", 'test': {'nested': "value"}})


class RemoveListElement(TypeSafeChainableFunction):
    class Param(TypeSafeChainableFunction.Param):
        l: List[Any]

    def __init__(self):
        super(RemoveListElement, self).__init__(name="remove_list_element", uuid="0003",
                                                param_class=RemoveListElement.Param)

    def func(self, param):
        print(type(param))
        print(param.l)
        del param.l[-1]
        print(param.l)


# m = TypeSafeChainableFunction.Meta()
# print(type(m))
# GetRaw.name
get_raw = GetRaw()
obj = ChainableObject()
obj = get_raw.apply(obj)
pprint(obj.dict())
get_raw2 = GetRawTypeSafe()
obj = ChainableObject()
obj = get_raw2.apply(obj, {'debug': True, 'data_name': "Test"})
pprint(obj.dict())
get_raw3 = GetRawFullTypeSafe()
obj = ChainableObject()
obj = get_raw3.apply(obj, {'debug': True, 'data_name': "Test"})
pprint(obj.dict())
# print(obj.meta['raw'][0].dict())
# pprint(obj.meta['raw'][0].data_class.schema())
# pprint(type(obj.meta['raw'][0].data_class))
mapping = {'param': {
    'param3': {'match': {'meta': {'jsonpath': 'meta.*[?name = "Test"]'}}, 'value': {'data': {'jsonpath': 'content'}}},
    # default: traverse to data branch
    'param4': {'match': {'meta': {'jsonpath': 'meta.*[?data_class_name = "GetRawFullTypeSafe.Data"]'}},
               'value': {'data': {'jsonpath': 'content'}}},  # value path relative to match path
}}
mapping = obj.resolve(mapping)
pprint(mapping)

Chainable.get_raw = GetRaw()
l1 = lambda: False
obj = ChainableObject()
obj.apply({
    'func': "Chainable.get_raw",
    'param': {'debug': False}
})
pprint(obj.dict())
mapping = {'param': {
    'debug1': True,
    'debug2': lambda: False,
    'debug3': {'static': False},
    # 'debug4': {'eval': "hist[-1]['func']['name'] == 'get_raw'"},
    'param1': {'eval': "data['raw'][0]"},
    'param2': {'jsonpath': 'meta.*[?name = "raw"].label.de'},  # eval jsonpath
    'param3': {'match': {'meta': {'jsonpath': 'meta.*[?name = "raw"]'}}, 'value': {'data': {'jsonpath': '[0]'}}},
    # default: traverse to data branch
    'param4': {'match': {'meta': {'jsonpath': 'meta.*[?name = "raw"]'}}, 'value': {'meta': {'jsonpath': 'label'}}},
    # value path relative to match path
}}
mapping = obj.resolve(mapping)
pprint(mapping)


def get_mapping():
    return {
        'func': "Chainable.get_raw",
        'param': {'debug': {'static': False}}
    }


# Chainable.get_raw = GetRaw()
Chainable.get_raw = GetRawTypeSafe()
Chainable.remove_list_element = RemoveListElement()
obj = ChainableObject()
obj = obj.apply({
    'func': "Chainable.get_raw",
    'param': {'debug': False}
}).apply(get_mapping())
pprint(obj.dict())

workflow = [{
    'func': "Chainable.get_raw",
    'param': {'debug': False}
}, {
    'func': "Chainable.get_raw",
    'param': {'debug': False, 'data_name': "RawVoltage"}
}, {
    'func': "Chainable.remove_list_element",
    'param': {
        'l': {'match': {'meta': {'jsonpath': 'meta.*[?name = "RawVoltage"]'}}, 'value': {'data': {'jsonpath': ''}}}}
}]
obj2 = ChainableObject()
for step in workflow:
    obj2 = obj2.apply(step)
pprint(obj2.dict())