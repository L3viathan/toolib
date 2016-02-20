#toolib

Whenever I write a function or class I feel I'll need at some point in the future, I'll put it here.

##Modules

`nlp` contains:

- `Tree` (and its child `PTree`) is in an implementation of simple parse trees. They can be created manually, or from either PTB or qtree format. PTree adds probabilities to the nodes.
- `ContextFreeLanguage`, which contains a CYK parser (and a method for estimating probabiities based on a PTB-like corpus).

##Classes

`DirectedGraph` is a naive implementation of a directed graph that was the result of a homework in a Python course I once did. It still works, but is in need of a rewrite.

`Heap` is a wrapper around a list with the `heapq` module of the standard library that accepts a key argument, which can be used to have heaps that are sorted by something else than the standard key.

`CSV` is a simple wrapper for csv files. Returns namedtuples, can write from tuples. Nicer than `csv.writer` and `csv.reader`, imho. May break with some files, though, so don't use in production.

##Functions

`nested_loop` is a simple function to avoid having to create deeply nested loops.

`eprint` is a convenience function that prints to stderr automatically.

`getchr` gets a single character from the terminal, without printing it. This gets the characters in sort of a raw format, caveats are explained in the docstring.

`parametrized`, and `fmap` are ways to give a decorator additional arguments. Please note that parametrized was not written by me.

`argmap` is a decorator that maps some function over all arguments, e.g. to turn them all to lowercase.

`runtime` is a decorator to run a function *n* times, and measure the time. The result of the last call is returned.

`nop` returns the first argument.

`fuzzy_match` returns the best match (if any) of an input string in a list of candidate strings. Look in the code to see what "match" means.

`most` is like the builtins `any` and `all`, in that it checks whether *most* of the iterable is truthy, and not like them in that it's kind of a joke.

`limit` yields the first couple of elements of an iterable (think `head`).
