import functools
import multiprocessing


class MyFancyClass(object):
    def __init__(self, name):
        self.name = name

    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print('Doing something fancy in %s for %s!' % (proc_name, self.name))
        return proc_name


def worker(requests, responses, class_args, class_kwargs):
    his_fancy_class = MyFancyClass(*class_args, **class_kwargs)
    func_name, func_args, func_kwargs = requests.get()
    responses.put(getattr(his_fancy_class, func_name)(*func_args, **func_kwargs))


def better_worker(requests, responses, factory):
    my_instance = factory()
    received = requests.get()
    while received != "poison":
        func_name, func_args, func_kwargs = received
        responses.put(getattr(my_instance, func_name)(*func_args, **func_kwargs))
        received = requests.get()


def make_dynamic(factory):
    requests = multiprocessing.Queue()
    responses = multiprocessing.Queue()
    p = multiprocessing.Process(target=better_worker, args=(requests, responses, factory))
    p.start()

    def call_remote(method_name, *args, **kwargs):
        requests.put((method_name, args, kwargs))
        return responses.get()

    class DynamicProxyProcess(object):
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
            return functools.partial(call_remote, method_name=s)

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            requests.put("poison")
            requests.close()
            requests.join_thread()
            responses.close()
            responses.join_thread()
            p.join()

    return DynamicProxyProcess()


def fancy_class_factory():
    return MyFancyClass("first_child")


def main():
    with make_dynamic(fancy_class_factory) as proxy:
        print("main received:", proxy.do_something())
        print("main received:", proxy.do_something())
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
