import functools
import multiprocessing


class MyFancyClass(object):
    def __init__(self, name):
        self.name = name

    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print('Doing something fancy in %s for %s!' % (proc_name, self.name))
        return proc_name

def better_worker(requests, responses, factory):
    my_instance = factory()
    received = requests.get()
    while received != "poison":
        func_name, func_args, func_kwargs = received
        responses.put(getattr(my_instance, func_name)(*func_args, **func_kwargs))
        received = requests.get()


class DynamicProxyProcess(object):
    def __init__(self, factory):
        self._responses = multiprocessing.Queue()
        self._requests = multiprocessing.Queue()
        self._p = multiprocessing.Process(target=better_worker, args=(self._requests, self._responses, factory))
        self._p.start()

    def _call_remote(self, method_name, *args, **kwargs):
        self._requests.put((method_name, args, kwargs))
        return self._responses.get()

    def __getattribute__(self, s):
        """
        this is called whenever any attribute of a NewCls object is accessed. This function first tries to
        get the attribute off NewCls. If it fails then it tries to fetch the attribute from self.oInstance (an
        instance of the decorated class). If it manages to fetch the attribute from self.oInstance, and
        the attribute is an instance method then `time_this` is applied.
        """
        try:
            x = super(DynamicProxyProcess, self).__getattribute__(s)
        except AttributeError:
            pass
        else:
            return x
        return functools.partial(self._call_remote, method_name=s)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._requests.put("poison")
        self._requests.close()
        self._requests.join_thread()
        self._responses.close()
        self._responses.join_thread()
        self._p.join()


def fancy_class_factory():
    return MyFancyClass("first_child")
def fancy_class_factory2():
    return MyFancyClass("second_child")


def main():
    with DynamicProxyProcess(fancy_class_factory) as proxy:
        with DynamicProxyProcess(fancy_class_factory2) as proxy2:
            print("main received from 1:", proxy.do_something())
            print("main received from 2:", proxy2.do_something())
            print("main received from 1:", proxy.do_something())

        # requests = multiprocessing.Queue()
        # responses = multiprocessing.Queue()
        # args = ["he"]
        # kwargs = {}
        # p = multiprocessing.Process(target=worker, args=(requests, responses, args, kwargs))
        # p.start()
        #
        # requests.put(("do_something", [], {}))
        # print("main received:", responses.get())
        # # Wait for the worker to finish
        # requests.close()
        # requests.join_thread()
        # p.join()


if __name__ == '__main__':
    main()
