"""L3viathan's universal module for stuff he thinks he'll need later."""

import heapq
import time
import sys
import csv
import termios
import tty
import operator
from functools import reduce
from collections import defaultdict, deque, namedtuple


class DirectedGraph:
    """Directed Graph, came from a homework solution."""

    def __init__(self):
        """Return an empty DirectedGraph object."""
        self.nodes = set()
        self.edges_out = defaultdict(set)
        self.edges_in = defaultdict(set)

    def add_edge(self, source, target):
        """
        Add an edge to the graph.

        source and target are either existing nodes, or newly created.
        Yes, that does sound like a bad idea, but I didn't make the homework.
        I guess I'll make the whole class more reasonable later.
        """
        self.edges_out[source].add(target)
        self.edges_in[target].add(source)
        self.nodes.add(source)
        self.nodes.add(target)

    def has_edge(self, source, target):
        """Return True iff there is an edge from source to target."""
        return target in self.edges_out[source]

    def __contains__(self, something):
        """Return True iff either something is a node or an edge."""
        if something in self.nodes:
            return True
        elif len(something) == 2:
            return self.has_edge(something[0], something[1])
        else:
            return False

    def edges(self):
        """Yield all edges."""
        for src in self.edges_out:
            for tgt in self.edges_out[src]:
                yield (src, tgt)

    def indeg(self, node):
        """Return the number of incoming edges of a node."""
        return len(self.edges_in[node])

    def outdeg(self, node):
        """Return the number of outgoing edges of a node."""
        return len(self.edges_out[node])

    def neighbors(self, node):
        """Return all neighbors of a node."""
        return union(self.edges_in[node], self.edges_out[node])

    def has_cycle(self):
        """Return True iff there is a cycle in the graph."""
        extended_edges = self.edges_out.copy()
        change = True
        while change:
            change = False
            for node in self.nodes:
                for tgt in extended_edges[node].copy():
                    for element in extended_edges[tgt]:
                        if element not in extended_edges[node]:
                            extended_edges[node].add(element)
                            change = True
        return any(x in extended_edges[x] for x in self.nodes)

    def dfs(self, node, func=deque.pop):
        """Yield all nodes in depth-first search."""
        visited = set()
        stack = deque([node])
        while stack:
            node = func(stack)
            if node not in visited:
                visited.add(node)
                yield node
                stack.extend(self.edges_out[node])

    def bfs(self, node):
        """Yield all nodes in breadth-first search."""
        for x in self.dfs(node, deque.popleft):
            yield x

    def tsort(self):
        """
        Yield all nodes in topological order, if possible.

        Otherwise, the behaviour is undefined.
        """
        active_nodes = self.nodes.copy()
        while active_nodes:
            for node in list(active_nodes):  # list() for the copy
                if all(x not in active_nodes for x in self.edges_in[node]):
                    yield node
                    active_nodes.remove(node)


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
                yield "_" + item.replace(" ", "")
            except:
                yield item.replace(" ", "")


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
    truthiness = 0
    for element in iterable:
        truthiness += 1 if element else -1
    return truthiness > 0


def limit(iterable, number=10):
    """Yield the top n elements of an iterable, like 'head' on unix."""
    for x, _ in zip(iterable, range(number)):
        yield x


def exhaust(iterator):
    """Exhaust an iterator."""
    deque(iterator, maxlen=0)

if __name__ == '__main__':
    with CSV(open("testfile.csv", "r")) as c:
        for line in c:
            print(line)
