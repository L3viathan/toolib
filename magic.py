import operator as op
from functools import partial

__all__ = ["ƒ", "functional_magic"]

class Magic(object):
    # TODO: implement methodcaller
    # Design decision (for now): only implement operators and ...getters, not
    # stuff like abs() or bool()
    def __init__(self, *, is_original=True):
        self.is_original = is_original
        self.__ops = []
        self.__repr = 'ƒ'
        self.__last_prec = 18

    def __repr__(self):
        return self.__repr

    def __call__(self, something):
        ''' Resolve the list of operations and return the final object. '''
        # Maybe eventually extend to act like methodcaller, but problems exist
        if self.is_original:
            raise RuntimeError("Can't call ƒ directly")
        for function in self.__ops:
            something = function(something)
        return something

    def _add_function(self, function, left='', right='', prec=0):
        '''
        Adds a function on the operation list.

        Additional arguments specify what strings should be added to the left
        or right side, and the operator precedence value of the operation.
        Parentheses are only added to the string representation if the new
        precedence value is higher than the previous one.

        If this is the one and only original ƒ object, make a non-original copy
        and call _add_function on it with the same parameters.
        '''
        if self.is_original:
            return type(self)(is_original=False)._add_function(
                function,
                left=left,
                right=right,
                prec=prec,
            )
        if prec > self.__last_prec:
            self.__repr = '{left}({previous}){right}'.format(
                left=left,
                right=right,
                previous=self.__repr,
            )
            self.__last_prec = prec
        else:
            self.__repr = '{left}{previous}{right}'.format(
                left=left,
                right=right,
                previous=self.__repr,
            )
            self.__last_prec = prec
        self.__ops.append(function)
        return self

    def __getattr__(self, attr):
        '''
        ƒ.foo

        If you want to use this with double underscore attributes, add a third
        underscore, e.g. ƒ.___name__
        '''
        if attr.startswith('___'):
            attr = attr[1:]
        elif attr.startswith('__'):
            raise AttributeError(attr)
        return self._add_function(op.attrgetter(attr), right='.'+attr, prec=16)

    def __getitem__(self, item):
        '''ƒ[foo]'''
        return self._add_function(op.itemgetter(item), right='['+repr(item)+']', prec=16)

    def __matmul__(self, other):
        '''Not actually matmul. ƒ@other. Applies function.'''
        return self._add_function(other, right='@'+other.__name__, prec=12)

    def __eq__(self, other):
        '''ƒ == other'''
        return self._add_function(partial(op.eq, other), right='=='+repr(other), prec=7)

    def __add__(self, other):
        '''ƒ + other'''
        return self._add_function(partial(op.add, other), right='+'+repr(other), prec=11)

    def __and__(self, other):
        ''' ƒ & foo '''
        return self._add_function(partial(op.and_, other), right='&'+repr(other), prec=9)

    def __mod__(self, other):
        ''' ƒ % foo '''
        return self._add_function(lambda x: x%other, right='%'+repr(other), prec=12)

    def __mul__(self, other):
        ''' ƒ * foo '''
        return self._add_function(partial(op.mul, other), right='*'+repr(other), prec=12)

    def __neg__(self):
        ''' -ƒ '''
        return self._add_function(partial(op.neg), left='-', prec=13)

    def __not__(self):
        ''' not ƒ'''
        return self._add_function(partial(op.not_), left='not ', prec=6)

    def __or__(self, other):
        ''' ƒ | foo '''
        return self._add_function(partial(op.or_, other), right='|'+repr(other), prec=8)

    def __pow__(self, other):
        ''' ƒ ** foo '''
        return self._add_function(lambda x: x**other, right='**'+repr(other), prec=14)

    def __sub__(self, other):
        ''' ƒ - foo '''
        return self._add_function(lambda x: x-other, right='-'+repr(other), prec=11)

    def __xor__(self, other):
        ''' ƒ ^ foo '''
        return self._add_function(partial(op.xor, other), right='-'+repr(other), prec=8)


ƒ = Magic()
functional_magic = ƒ
