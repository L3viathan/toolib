#toolib

Whenever I write a function or class I feel I'll need at some point in the future, I'll put it here.

##Classes

`Tree` (and its child `PTree`) is in an implementation of simple parse trees. They can be created manually, or from either PTB or qtree format. PTree adds probabilities to the nodes.

`DirectedGraph` is a naive implementation of a directed graph that was the result of a homework in a Python course I once did. It still works, but is in need of a rewrite.

`Heap` is a wrapper around a list with the `heapq` module of the standard library that accepts a key argument, which can be used to have heaps that are sorted by something else than the standard key.

##Functions

`nested_loop` is a simple function to avoid having to create deeply nested loops.

`eprint` is a convenience function that prints to stderr automatically.

`getchr` gets a single character from the terminal, without printing it. This gets the characters in sort of a raw format, caveats are explained in the docstring.

`parametrized`, and `fmap` are ways to give a decorator additional arguments. Please note that parametrized was not written by me.

`argmap` is a decorator that maps some function over all arguments, e.g. to turn them all to lowercase.

`nop` returns the first argument.
