import multiprocessing
from threading import Lock


class _MyFancyClass(object):
    def __init__(self, name):
        self.name = name

    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print('Doing something fancy in %s for %s!' % (proc_name, self.name))
        return proc_name


def _better_worker(connection, factory):
    my_instance = factory()
    try:
        while True:
            func_name, func_args, func_kwargs = connection.recv()
            # This is a poison pill
            if func_name is None:
                break
            try:
                return_value = getattr(my_instance, func_name)(*func_args, **func_kwargs)
            except KeyboardInterrupt:
                print("keyboard interupt in proxy worker process")
                break
            except Exception as e:
                connection.send(e)
            else:
                if return_value is None:
                    return_value = "__None__"
                connection.send(return_value)
    except KeyboardInterrupt:
        # the keyboard interrupt can happen in the inner try block or in the recv
        print("keyboard interupt in proxy worker process")


class DynamicProxyProcess(object):
    def __init__(self, factory):
        self._parent_con, self._child_con = multiprocessing.Pipe(duplex=True)
        self._lock = Lock()
        self._p = multiprocessing.Process(target=_better_worker, args=(self._child_con, factory))
        self._p.daemon = True
        self._p.start()

    def _make_remote_call(self, method_name):
        def call_remote(*args, **kwargs):
            with self._lock:
                self._parent_con.send((method_name, args, kwargs))
                return self._parent_con.recv()

        return call_remote

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
        return self._make_remote_call(s)

    # def __del__(self):
    #     self._close()
    #
    # def __enter__(self):
    #     return self
    #
    def _close(self):
        # I don't think this is a 100% reliable but it works for now
        # with propergation of Keyboard interupt signals and daemon processes I don't even if poison pills are needed -FM
        # with self._lock:
            self._parent_con.send((None, None, None))
            self._p.join()

            # def __exit__(self, type, value, traceback):
            #     self._close()


def fancy_class_factory():
    return _MyFancyClass("first_child")


def fancy_class_factory2():
    return _MyFancyClass("second_child")


def main():
    with DynamicProxyProcess(fancy_class_factory) as proxy:
        with DynamicProxyProcess(fancy_class_factory2) as proxy2:
            print("main received from 1:", proxy.do_something())
            print("main received from 2:", proxy2.do_something())
            print("main received from 1:", proxy.do_something())
    loos_proxy = DynamicProxyProcess(fancy_class_factory())


if __name__ == '__main__':
    main()
