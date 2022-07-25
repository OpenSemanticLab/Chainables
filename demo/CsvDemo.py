import sys
sys.path.append('../')

from src.chainables import Chainable, ChainableObject, ChainableFunction, TypeSafeChainableFunction
from typing import List, Optional, Any, Union
from pprint import pprint

import os
import io
import chardet
import csv

class Files:
    pass

Chainable.Files = Files()

class ReadTextFile(TypeSafeChainableFunction):
    class Param(TypeSafeChainableFunction.Param):
        path: str
        encoding: Optional[str] = 'auto'

    class Data(TypeSafeChainableFunction.Data):
        text: str

    class Meta(TypeSafeChainableFunction.Meta):
        path: str
        encoding: str
        info: os.stat_result

    def __init__(self):
        super(__class__, self).__init__(name="read_text_file", uuid="ab61cbad-cd75-4634-85ff-6eab39bd960f",
                                        param_class=__class__.Param, data_class=__class__.Data,
                                        meta_class=__class__.Meta)

    def func(self, param: Param):
        info = os.stat(param.path)
        if (param.encoding == 'auto'):
            #see https://stackoverflow.com/questions/436220/how-to-determine-the-encoding-of-text
            with open(param.path, 'rb') as f:
                rawdata = b''.join([f.read() for _ in range(20)]) #read only the first 20 chars
                param.encoding = chardet.detect(rawdata)['encoding']
        with io.open(param.path, mode="r", encoding=param.encoding) as f:
            text = f.read()
        super().store('text_file',
                      {'text': text, 'info': info},
                      {'path': param.path,
                       'encoding': param.encoding,
                       'info': info
                       }
                      )

Chainable.Files.read_text_file = ReadTextFile()

class ParseCsvText(TypeSafeChainableFunction):
    class Param(TypeSafeChainableFunction.Param):
        text: str
        delimiter: Optional[str] = "auto"

    class Data(TypeSafeChainableFunction.Data):
        data: dict

    class Meta(TypeSafeChainableFunction.Meta):
        sep: str

    def __init__(self):
        super(__class__, self).__init__(name="parse_csv_text", uuid="ac5040ca-547b-4d52-bff2-62d5cfcc692c",
                                        param_class=__class__.Param, data_class=__class__.Data,
                                        meta_class=__class__.Meta)

    def func(self, param: Param):
        if (param.delimiter == 'auto'):
            #see https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters
            dialect = csv.Sniffer().sniff(param.text)
            param.delimiter = dialect.delimiter
        data = {}
        super().store('csv_data',
                      {'data': data},
                      {'sep': param.delimiter
                       }
                      )

Chainable.Files.parse_csv_text = ParseCsvText()

obj = ChainableObject()
lang = 'en'
obj = obj.apply({
    'func': "Chainable.Files.read_text_file",
    'param': {'debug': False, 'path': r"csv\files\Anton-Paar_RheoCompas_Export_simple.csv", 'encoding': "auto"}
}).apply({
    'func': "Chainable.Files.parse_csv_text",
    'param': {'debug': False, 'text': {'eval': "data['text_file'][0].text"}, 'delimiter': "auto"}
})
pprint(obj.dict())