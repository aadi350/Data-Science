def typetest(**argchecks):
    def wrapper(func):
        def on_call(*args, **kwargs):
            for (argname, type) in argchecks.items():
                if argname in kwargs:
                    if not isinstance(kwargs[argname], type):
                        raise TypeError(errmsg)
                elif argname in args:
                    position = args.index(argname)
                    if not isinstance(args[position], type):
                        raise TypeError(errmsg)
                else:
                    # Assume not passed: default
                    pass
            return func(*args, **kwargs)
        return on_call
    return wrapper
