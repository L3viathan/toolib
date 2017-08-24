"""L3viathan's universal module for stuff he thinks he'll need later."""

import heapq
import time
import sys
import csv
import termios
import tty
import operator
import random
from functools import reduce
from collections import defaultdict, deque, namedtuple
from itertools import zip_longest


class Heap(list):
    """
    A lightweight heap essentially utilizing the heapq module.

    It can however be supplied a key argument on initialization. The heap
    itself is a list of tuples of (key, element), but popping is transparent.
    """

    def __init__(self, initial=None, key=lambda x: x):
        """
        Return an empty heap.

        If it has the argument 'initial', it is assumed to be an iterable from
        which the heap will be initialized.
        'key' is a function similar to those usable in the sort() function,
        which will be used whenever a comparison is made.
        """
        self.key = key
        if initial:
            self.extend((key(item), item) for item in initial)
            heapq.heapify(self)

    def push(self, item):
        """Push an element on the heap."""
        heapq.heappush(self, (self.key(item), item))

    def pop(self):
        """Pop the smallest element off the heap, maintaining the invariant."""
        return heapq.heappop(self)[1]

    def replace(self, item):
        """
        Pop an element off the heap, then push.

        More efficient then first popping and then pushing.
        """
        return heapq.heapreplace(self, (self.key(item), item))[1]

    def pushpop(self, item):
        """
        Push an element on the heap, then pop and return the smallest item.

        More efficient then first pushing and then popping.
        """
        return heapq.heappushpop(self, (self.key(item), item))[1]


class CSV(object):
    """
    Simple CSV wrapper for a file-like object.

    Lines will be returned as a namedtuple with the column names infered from
    the header.
    When used as a context manager, it will automatically close the file object
    on exiting the context.
    Object supports reading or writing, depending on whether the file object
    does.

    Usage:
        with CSV(open(...)) as f:
            for line in f:
                pass
    """

    def __init__(self, fileobject, **kwargs):
        """Initialize a CSV object.

        Optional keyword arguments:
            delimiter: The seperation character (default: ',')
            header: Whether the CSV file has a header (default: True)
        """
        self.fileobject = fileobject
        if self.fileobject.readable():
            arguments = {
                    "delimiter": kwargs.get("delimiter", ","),
                    "quotechar": kwargs.get("quotechar", '"'),
                    }
            self.reader = csv.reader(self.fileobject, **arguments)
            if kwargs.get("header", True):
                data = list(CSV._makeidentifier(next(self.reader)))
                self.Row = namedtuple("Row", data)
            else:
                self.Row = None  # get later
        elif self.fileobject.writable():
            arguments = {
                    "delimiter": kwargs.get("delimiter", ","),
                    "quotechar": kwargs.get("quotechar", '"'),
                    }
            self.writer = csv.writer(self.fileobject, **arguments)
        else:
            raise OSError("File neither readable nor writable")

    def readline(self):
        """Return a namedtuple."""
        if self.Row is None:
            data = next(self.reader)
            self.Row = namedtuple(
                    "Row",
                    ["col" + str(i) for i, _ in enumerate(data)],
                    )
            return self.Row(*data)
        else:
            return self.Row(*next(self.reader))

    def read(self):
        """Return a list of namedtuples."""
        lines = []
        for line in self:
            lines.append(line)
        return lines

    def __iter__(self):
        """Return self, because __next__ handles iteration."""
        return self

    def __next__(self):
        """Magic method for iteration."""
        return self.readline()

    def write(self, line):
        """Write a single line to the file."""
        self.writer.writerow(line)

    def __enter__(self):
        """Context manager entry: Do nothing."""
        return self

    def __exit__(self, *args):
        """Close file object when leaving context."""
        self.fileobject.close()

    @staticmethod
    def _makeidentifier(some_list):
        """Lazy way to make columns behave. May not always work."""
        for item in some_list:
            try:
                int(item[0])
                yield "_" + item.replace(" ", "").replace("-", "_")
            except:
                yield item.replace(" ", "").replace("-", "_")


def parametrized(dec):
    """Decorate decorators with parameters.

    A decorator for decorators that turns them into functions taking
    arguments and returning said decorator without the additional parameters.
    In short, this allows your decorators to take arguments.
    For a different solution to the same problem, see fmap.

    Stolen from http://stackoverflow.com/users/282614/dacav
    """
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


def fmap(fn, *args, **kwargs):
    """
    Map a function with arguments to a decorator without arguments.

    Takes a function, and optionally some arguments, and returns it as a
    decorator (without parameters). This has the same effect as parametrized,
    but can be done ad-hoc without needing to decorate the decorator. This is
    useful for unintended decorators (functions that happen to take a function
    and return another, but weren't meant to be used as decorators).

    @fmap(some_function, arg1, arg2, arg3=val3)
    def foobar(...):
        ...
    """
    def decorator(inner_fn):
        return fn(inner_fn, *args, **kwargs)
    return decorator


def nested_loop(n, l):
    """
    Return n-tuples counting up range(l) individually.

    As should be obvious from the name, this is a replacement for deeply
    nested loops. If you consider using it, first consider not using it.

    It turns:
    for x in range(3):
        for y in range(3):
            print(x, y)

    into:

    for x, y in nested_loops(2, 3):
        print(x, y)
    """
    for c in range(l ** n):
        yield tuple(c // l**x % l for x in reversed(range(n)))


def argmap(map_fn):
    """
    Decorate function to perform some transformation on the arguments.

    Example usage:
    @argmap(my_function, 1, 2, 3)
    def foo(bar):
        pass

    becomes:

    ...
    foo = my_function(foo, 1, 2, 3)
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*map(map_fn, args), **kwargs)
        return wrapper
    return decorator


def runtime(how_many_tries=10):
    """Decorate function to measure its runtime."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            eprint("Timing {} with {} tries".format(
                fn.__name__,
                how_many_tries,
                ))
            times = []
            ret = None  # just in case how_many_tries is set to 0
            for i in range(how_many_tries):
                beginning = time.time()
                ret = fn(*args, **kwargs)
                runtime = time.time() - beginning
                times.append(runtime)
                eprint("Run {0} completed in {1:.3f} seconds".format(
                    i+1,
                    runtime,
                    ))
            eprint("All runs done. Average time: {0:.3f}".format(
                sum(times)/len(times)
                ))
            return ret
        return wrapper
    return decorator


def eprint(*args, **kwargs):
    """Print to stderr. Convenience function."""
    print(*args, file=sys.stderr, **kwargs)


def nop(something):
    """
    Return the first argument completely unchanged.

    This is the third way to solve the generators-with-arguments problem.
    Since you can't use lambda expressions as decorators, this is a way around
    that. You can just wrap the lambda expression in a nop()-call.

    @nop(lambda x: some_generator_with_args(x, arg1, ...))
    def foo():
        pass
    """
    return something


def getchr():
    r"""
    Get a single key from the terminal without printing it.

    Certain special keys return several "characters", all starting with the
    escape character '\x1b'. You could react to that by reading two more
    characters, but the actual escape key will only make this return a single
    escape character, and since it's blocking, that wouldn't be a good solution
    either.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ord(ch) == 3:  # ^C
        raise KeyboardInterrupt
    return ch


def fuzzy_match(req, ls):
    """
    Given a list of candidates and a partial string, return the 'best' match.

    >>> l = ["apple", "orange", "banana", "pear"]
    >>> fuzzy_match("p", l)
    pear
    >>> fuzzy_match("pp", l)
    apple
    >>> fuzzy_match("oe", l)
    orange

    """
    if req == "":
        return False
    for candidate in ls:
        reqc = req
        formatted = ""
        for char in candidate:
            if len(reqc) and char.lower() == reqc[0]:
                reqc = reqc[1:]
                formatted += char
            else:
                formatted += char
        if len(reqc) == 0:
            return candidate
    return False


def most(iterable):
    """Return True if 'most' elements are truthy."""
    truthiness = [*map(bool, iterable)]
    return sum(truthiness) > (len(truthiness)//2)


def head(iterable, number=10):
    """Yield the top n elements of an iterable, like 'head' on unix."""
    for x, _ in zip(iterable, range(number)):
        yield x

def tail(iterable, number=10):
    """Yield the last n elements of an iterable, like 'tail' on unix."""
    t = deque(maxlen=number)
    for x in iterable:
        t.append(x)
    yield from t


def exhaust(iterator):
    """Exhaust an iterator."""
    deque(iterator, maxlen=0)


def pairwise(sequence):
    """Iterate over pairs of a sequence."""
    i = iter(sequence)
    j = iter(sequence)
    next(j)
    yield from zip(i, j)


def nwise(iterable, n):
    """
    Iterate over n-grams of an iterable.

    Has a bit of an overhead compared to pairwise (although only during
    initialization), so the two functions are implemented independently.
    """
    iterables = [iter(iterable) for _ in range(n)]
    for index, it in enumerate(iterables):
        for _ in range(index):
            next(it)
    yield from zip(*iterables)


def weighted_choice(choices):
    """
    Weighted version of random.choice.
    Takes a dictionary of choices to weights. Weights can be percentages or
    any kind of weigth, they will be normalized.
    """
    N = sum(choices.values())
    rand = random.random()
    for candidate, weight in choices.items():
        if rand <= weight/N:
            return candidate
        else:
            rand -= weight/N


def multimap(iterable, *maps):
    """
    Map a different function to each element of an iterable.

    One of the usecases is when parsing some string with different types
    "inside". Normally, you would have to do:
    >>> a, b, c, d = "13,Haha,True,-3".split(",")
    >>> a, c, d = int(a), bool(c), int(d)
    >>> a, b, c, d
    (13, 'Haha', True, -3)

    Using multimap, you can instead do:
    >>> a, b, c, d = multimap("13,Haha,True,-3".split(","), int, None, bool, int)
    >>> a, b, c, d
    (13, 'Haha', True, -3)

    If None is given as a function, multimap leaves the corresponding element
    untouched. If you give fewer functions than items in the iterable, the last
    ones will be untouched as well.
    """
    for fn, element in zip_longest(maps, iterable):
        if fn is not None:
            yield fn(element)
        else:
            yield element

if __name__ == '__main__':
    with CSV(open("testfile.csv", "r")) as c:
        for line in c:
            print(line)
