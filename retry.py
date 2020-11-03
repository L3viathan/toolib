import random
import functools

def retry(tries):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ex = None
            for i in range(tries):
                try:
                    val = fn(*args, **kwargs)
                except Exception as e:
                    ex = e
                else:
                    return val
            if ex is not None:
                raise ex
        return wrapper
    return decorator


if __name__ == '__main__':
    @retry(3)
    def fail_often():
        if random.random() > 0.9:
            return "yay"
        raise RuntimeError


    fail_often()
