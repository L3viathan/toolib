import random
import functools

def retry(tries):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ex = None
            for i in range(tries):
                try:
                    print("Try nr.", i)
                    val = fn(*args, **kwargs)
                    print("Success!")
                except Exception as e:
                    print("Failed")
                    ex = e
                else:
                    return val
            if ex is not None:
                print("Max. tries exceeded, reraising")
                raise ex
        return wrapper
    return decorator


@retry(3)
def fail_often():
    if random.random() > 0.9:
        return "yay"
    raise RuntimeError


fail_often()
