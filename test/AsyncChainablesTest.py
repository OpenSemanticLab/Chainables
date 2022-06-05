import asyncio
import sys
sys.path.append('../')

from src.chainables import Chainable, ChainableObject, ChainableFunction, TypeSafeChainableFunction
from src.chainables import AsyncChainableObject, AsyncChainableContext, AsyncChainableEventFunction
from pprint import pprint
from time import sleep

class AsyncChainableFunction(ChainableFunction):
    def __init__(self):
        super(__class__, self).__init__(name="async_function", uuid="0004")

    def func(self, param):
        sleep(1)
        print(param['msg'])
        if 'print_obj' in param and param['print_obj']: pprint(self.obj.dict())

Chainable.async_log = AsyncChainableFunction()


class EventGenerator(AsyncChainableEventFunction):
    def __init__(self):
        super(__class__, self).__init__(name="event_generator", uuid="0005")

    async def loop(self, param):
        timer = 0
        while (timer < 3):
            await asyncio.sleep(1)
            # sleep(1)
            # self.obj.store_data(
            self.emit()
            timer += 1
        print("Event loop done")

Chainable.async_event = EventGenerator()

async def test1():
    obj = AsyncChainableObject()
    obj.apply_parallel([
        {
            'func': 'Chainable.async_log', 'param': {'msg': 'test1'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test2'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test3'}
        }
    ]).done = lambda obj : obj.apply({
        'func': 'Chainable.async_log', 'param': {'msg': 'test9', 'print_obj': False}
    }).apply({
        'func': 'Chainable.async_log', 'param': {'msg': 'test10', 'print_obj': False}
    })

async def test2():
    obj = AsyncChainableObject()
    await (await (await obj.apply_parallel_async([
        {
            'func': 'Chainable.async_log', 'param': {'msg': 'test1'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test2'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test3'}
        }
    ])).apply_sequential_async([
        {
            'func': 'Chainable.async_log', 'param': {'msg': 'test4'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test5'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test6'}
        }
    ])).apply_parallel_async([
        {
            'func': 'Chainable.async_log', 'param': {'msg': 'test7'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test8'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test9'}
        }
    ])
    pprint(obj.dict())

async def test3():
    obj2 = AsyncChainableObject()
    context = AsyncChainableContext()
    obj2 = await context.run_async([
        [
            {
                'func': 'Chainable.async_log', 'param': {'msg': 'test1'}
            },{
                'func': 'Chainable.async_log', 'param': {'msg': 'test2'}
            },{
                'func': 'Chainable.async_log', 'param': {'msg': 'test3'}
            }
        ],
        {
            'func': 'Chainable.async_log', 'param': {'msg': 'test4'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test5'}
        },{
            'func': 'Chainable.async_log', 'param': {'msg': 'test6'}
        },
        [
            {
                'func': 'Chainable.async_log', 'param': {'msg': 'test7'}
            },{
                'func': 'Chainable.async_log', 'param': {'msg': 'test8'}
            },{
                'func': 'Chainable.async_log', 'param': {'msg': 'test9'}
            }
        ]
    ])
    pprint(obj2.dict())

async def test4():
#def test4():
    obj = AsyncChainableObject()
    # obj = EmitableChainableObject()
    print(type(obj))
    #obj.emit = lambda obj: obj.apply({
    #    'func': 'Chainable.async_log', 'param': {'msg': 'test9', 'print_obj': True}
    #})
    obj.apply({
        'func': 'Chainable.async_event', 'param': {'msg': 'test7'}
    }).emit = lambda obj: obj.apply({
        'func': 'Chainable.async_log', 'param': {'msg': 'test9', 'print_obj': True}
    })
    while(True):
        await asyncio.sleep(1)
        #print("Main")

#asyncio.run(test1())
#asyncio.run(test2())
#asyncio.run(test3())
#loop = asyncio.get_event_loop()
asyncio.get_event_loop().run_until_complete(test4())

