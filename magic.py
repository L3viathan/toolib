import operator as op
from functools import partial

class Magic(object):
    # TODO: implement methodcaller
    # Design decision (for now): only implement operators and ...getters, not
    # stuff like abs() or bool()
    def __init__(self, *, is_original=True):
        self.is_original=is_original
        self.__ops = []
        self.__repr = 'ƒ'

    def __repr__(self):
        return self.__repr

    def __call__(self, something):
        ''' Maybe eventually extend to act like methodcaller, but problems exist'''
        if self.is_original:
            # return Magic(is_original=False)
            raise RuntimeError("Can't call getter directly")
        for function in self.__ops:
            something = function(something)
        return something

    def _add_function(self, function, left='', right=''):
        if self.is_original:
            return type(self)(is_original=False)._add_function(
                    function,
                    left=left,
                    right=right,
                    )
        if self.__ops:
            self.__repr = '{left}({previous}){right}'.format(
                    left=left,
                    right=right,
                    previous=self.__repr,
                    )
        else:
            self.__repr = '{left}{previous}{right}'.format(
                    left=left,
                    right=right,
                    previous=self.__repr,
                    )
        self.__ops.append(function)
        return self

    def __getattr__(self, attr):
        '''
        getter.foo

        If you want to use this with double underscore attributes, add a third
        underscore, e.g. ƒ.___name__
        '''
        if attr.startswith('___'):
            attr = attr[1:]
        elif attr.startswith('__'):
           raise AttributeError(item)
        return self._add_function(op.attrgetter(attr), right='.'+attr)

    def __getitem__(self, item):
        '''getter[foo]'''
        return self._add_function(op.itemgetter(item), right='['+repr(item)+']')

    def __matmul__(self, other):
        '''Not actually matmul. getter@function. Applies function.'''
        return self._add_function(other, right='@'+other.__name__)

    def __eq__(self, other):
        return self._add_function(partial(op.eq, other), right='=='+repr(other))

    def __add__(self, other):
        return self._add_function(partial(op.add, other), right='+'+repr(other))

    def __and__(self, other):
        ''' ƒ & foo '''
        return self._add_function(partial(op.and_, other), right='&'+repr(other))

    def __mod__(self, other):
        ''' ƒ % foo '''
        return self._add_function(lambda x: x%other, right='%'+repr(other))

    def __mul__(self, other):
        ''' ƒ % foo '''
        return self._add_function(partial(op.mul, other), right='*'+repr(other))

    def __neg__(self):
        ''' -ƒ '''
        return self._add_function(partial(op.neg, other), left='-')

    def __not__(self):
        ''' not ƒ'''
        return self._add_function(partial(op.not_, other), left='not ')

    def __or__(self, other):
        ''' ƒ | foo '''
        return self._add_function(partial(op.or_, other), right='|'+repr(other))

    def __pow__(self, other):
        ''' ƒ ** foo '''
        return self._add_function(lambda x: x**other, right='**'+repr(other))

    def __sub__(self, other):
        ''' ƒ - foo '''
        return self._add_function(lambda x: x-other, right='-'+repr(other))

    def __xor__(self, other):
        ''' ƒ ^ foo '''
        return self._add_function(partial(op.xor, other), right='-'+repr(other))


ƒ = Magic()
functional_magic = ƒ
