"""Natural Language Processing tools."""

import re
import operator
from collections import defaultdict, Counter
from itertools import combinations
from functools import reduce


class ParseError(Exception):
    """Raised when a parse error occurs."""

    pass


class Tree(object):
    """Parse tree."""

    def __init__(self, label):
        """Return an empty tree with a given label."""
        self.label = label
        self.children = []

    @property
    def size(self):
        """Return the number of nodes in the subtree."""
        return 1 + sum(child.size for child in self.children)

    __len__ = size

    @property
    def depth(self):
        """Return the depth."""
        return 1 + max((child.depth for child in self.children), default=0)

    @property
    def leaf(self):
        """Return True iff we're a leaf."""
        return len(self.children) == 0

    @staticmethod
    def _tokenize(string):
        """Simple tokenization helper function for from_string."""
        # reversed because popleft isn't O(1)
        return list(reversed(re.findall(r'\(|\)|[^ \n\t()]+', string)))

    @classmethod
    def from_qtree(cls, string):
        r"""Take something like "\Tree [.S [.NP Peter ] ]" and return a tree."""
        string = (string
                  .replace("\\Tree ", "")
                  .replace("[", "(")
                  .replace("]", ")")
                  .replace(".", "")
                  )
        return cls.from_string(string)

    @classmethod
    def from_string(cls, string):
        """Take something like "(S (NP Peter))" and returns a tree."""
        tokens = cls._tokenize(string)
        return cls._tree(tokens)

    @classmethod
    def _tree(cls, tokens):
        """
        Build a single tree from the token list.

        To do that, it first checks whether the current token is a leaf (in
        which case it just returns that simple tree), and if not, creates a new
        tree with the next token as a label and then calls _trees, which yields
        trees until it sees the closing paranthesis in the current level. In
        that case we're done adding children and can return the tree.
        """
        t = tokens.pop()
        # if t is a paranthesis, we need to nest
        if t == '(':
            tree = cls(tokens.pop())
            for subtree in cls._trees(tokens):
                tree.children.append(subtree)
            return tree
        else:
            return cls(t)

    @classmethod
    def _trees(cls, tokens):
        """
        Helper method for from_string.

        This assumes we're inside a "list" of trees right now, and calls
        _tree to build a tree until it sees a closing paranthesis. Raising
        StopIteration will cause the for loop in _tree to terminate.
        """
        while True:
            if not tokens:
                raise StopIteration
            if tokens[-1] == ')':
                tokens.pop()
                raise StopIteration
            yield cls._tree(tokens)

    def parentize(self):
        for child in self.children:
            child.parent = self
            child.parentize()

    def treeview(self, prefix=""):
        ret = prefix
        if self.parent is not None:
            if self is self.parent.children[-1]:
                ret += "└─"
            else:
                ret += "├─"
        ret += self.label + ("\n" if self.children else "")
        if self.parent is None:
            ret += "\n".join(child.treeview(prefix) for child in self.children)
        elif self is not self.parent.children[-1]:
            ret += "\n".join(child.treeview(prefix+"│ ") for child in self.children)
        else:
            ret += "\n".join(child.treeview(prefix+"  ") for child in self.children)
        return ret

    def __str__(self):
        """
        Return the PTB notation of the tree.

        In addition, return the sentence level, i.e. all leaves.
        """
        self.parentize()
        return self.treeview()
        return """{}\n{}\n""".format(repr(self), self.text)

    def __repr__(self):
        """Return something like "Tree.from_string('(S (NP Peter))')"."""
        return "{}.from_string('{}')".format(type(self).__name__, self._repr())

    def _repr(self):
        """
        Recursive helper function for __repr__.

        Because we want the wrapper "Tree.from_string" only on the top level.
        """
        if self.leaf:
            return self.label
        else:
            kids = ' '.join(child._repr() for child in self.children)
            return '({} {})'.format(self.label, kids)

    def __eq__(self, other):
        """Check for equality."""
        return repr(self) == repr(other)

    def __format__(self, _):
        """Represent the tree in new-style string formatting."""
        return self._repr()

    def qtree(self):
        r"""Return something like \Tree [.S [.NP Peter ] [.VP sleeps ] ]."""
        return "\\Tree " + self._qtree()

    def _qtree(self):
        """Export in qtree format."""
        if self.leaf:
            return self.label
        else:
            kids = ' '.join(child._qtree() for child in self.children)
            return '[.{} {}]'.format(self.label, kids)

    def walk(self):
        """Yield itself and then recursively all decendents. DFS."""
        yield self
        for child in self.children:
            # this is a cool Python 3 feature: yield from an iterator
            yield from child.walk()

    @property
    def text(self):
        """The text of the tree; all leaves joined by a space."""
        return ' '.join(leaf.label for leaf in self.walk_leaves())

    def walk_leaves(self):
        """Yield only the leaves."""
        if self.leaf:
            yield self
        else:
            for child in self.children:
                yield from child.walk_leaves()

    def attach(self, other):
        """Put the other tree as the last child in this tree."""
        self.children.append(other)

    def empty(self):
        """Empty the children list, turning the node into a leaf."""
        # actually emptying the list
        del self.children[:]
    __iter__ = walk

    def __getitem__(self, index):
        """Convenience function for accessing children."""
        return self.children[index]

    def __deepcopy__(self):
        """Return a deep copy of the tree."""
        return type(self).from_string(self._repr())

    def __getattr__(self, name):
        """More usability. If there's only one node of the name, return it."""
        ret = None
        for child in self.children:
            if child.label == name:
                if ret:
                    raise AttributeError("More than one node of type " + name)
                ret = child
        return ret


class PTree(Tree):
    """Parse tree with probabilities."""

    def __init__(self, label, prob=1):
        """
        Return a parse tree with probabilities.

        The optional argument 'prob' can be set to a value between 0 and 1 to
        assign a probability to this node.
        """
        super().__init__(label)
        self.prob = float(prob)

    @property
    def probability(self):
        """Return the overall probability of the tree."""
        return reduce(operator.mul, (node.prob for node in self.walk()), 1)

    def __repr__(self):
        """Return something like "PTree.from_string('(S (NP Peter))')"."""
        return "{}.from_string('{}')".format(type(self).__name__, self._repr())

    def _repr(self):
        """
        Recursive helper function for __repr__.

        Because we want the wrapper "Tree.from_string" only on the top level.
        """
        if self.prob != 1:
            annotated_label = self.label + ":" + str(self.prob)
        else:
            annotated_label = self.label

        if self.leaf:
            return annotated_label
        else:
            kids = ' '.join(child._repr() for child in self.children)
            return '({} {})'.format(annotated_label, kids)

    @classmethod
    def _tree(cls, tokens):
        """
        Build a single tree from the token list.

        To do that, it first checks whether the current token is a leaf (in
        which case it just returns that simple tree), and if not, creates a new
        tree with the next token as a label and then calls _trees, which yields
        trees until it sees the closing paranthesis in the current level. In
        that case we're done adding children and can return the tree.
        """
        t = tokens.pop()
        # if t is a paranthesis, we need to nest
        if t == '(':
            tree = cls(*tokens.pop().split(":"))
            for subtree in cls._trees(tokens):
                tree.children.append(subtree)
            return tree
        else:
            return cls(*t.split(":"))


class ContextFreeLanguage(object):
    """Model of a context-free language."""

    def __init__(self, **kwargs):
        """
        Context-free language parser (mostly).

        Keyword arguments:

        default_POS:    default POS tag that will be assigned if POS_hook is
                        not defined. Automatically set when grammar is learned
                        using learn_grammar.
        """
        self._rule_counts = defaultdict(Counter)
        self._lex_counts = defaultdict(Counter)
        self.rules = defaultdict(dict)
        self.lexicon = defaultdict(dict)
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def learn_grammar(self, fileobject):
        """
        Given a file-like object containing PTB-like trees, learn a grammar.

        The trees should be in Chomsky-Normal-Form, although non-terminal unary
        rules will work.
        """
        self.POS_collection = defaultdict(set)
        for tree in Tree._trees(Tree._tokenize(fileobject.read())):
            self.count_tree(tree)

        for key in self._rule_counts:
            total = sum(self._rule_counts[key].values())
            for children in self._rule_counts[key]:
                self.rules[key][children] = self._rule_counts[key][children] / total

        for key in self._lex_counts:
            total = sum(self._lex_counts[key].values())
            for label in self._lex_counts[key]:
                self.lexicon[key][label] = self._lex_counts[key][label] / total

        self.default_POS = max(
                self.POS_collection,
                key=lambda x: len(self.POS_collection[x]),
                )

    def count_tree(self, tree):
        """Given a tree object, add it to the statistics."""
        for node in tree.walk():
            if node.leaf:
                continue
            elif node.children[0].leaf:
                self._lex_counts[node.children[0].label][node.label] += 1
                self.POS_collection[node.label].add(node.children[0].label)
            elif len(node.children) == 2:
                self._rule_counts[node.label][
                        node.children[0].label,
                        node.children[1].label,
                        ] += 1
            elif len(node.children) == 1:
                # unary rules. Not in CNF, but maybe acceptable.
                self._rule_counts[node.label][node.children[0].label] += 1
            else:
                raise ParseError("Trees must be in Chomsky-Normal Form "
                                 "(found node with more than 2 children)")

    def parse(self, sentence):
        """
        Given a sentence, return a list of parse trees.

        This uses the CYK parsing algorithm.
        """
        def walk_chart(length):
            for i in range(1, length):
                for j in range(length-i):
                    yield i, j

        def pairings(iterable1, iterable2):
            for item1 in iterable1:
                for item2 in iterable2:
                    yield item1, item2

        def get_rules(rules, arity=2):
            for key, rule_items in self.rules.items():
                for kids in rule_items:
                    if len(kids) == arity:
                        yield key, kids, rule_items[kids]

        chart = defaultdict(list)
        sentence = sentence.split()
        for index, word in enumerate(sentence):
            if self.lexicon[word]:
                for label in self.lexicon[word]:
                    tree = PTree(label, self.lexicon[word][label])
                    leaf = PTree(word)
                    tree.children = [leaf]
                    chart[(index, index)].append(tree)
            else:
                tree = PTree(self.POS_hook(word), self.lexicon[word][label])
                leaf = PTree(word)
                tree.children = [leaf]
                chart[(index, index)].append(tree)
            for element in chart[index, index]:
                for key, kids, prob in get_rules(self.rules, arity=1):
                    if kids[0] == element.label:
                        tree = PTree(key, prob)
                        tree.children = [element]
                        chart[index, index].append(tree)

        length = len(sentence)
        for i, j in walk_chart(length):
            for k in range(i):
                for left, right in pairings(chart[j, j+k], chart[j+1+k, i+j]):
                    for key, kids, prob in get_rules(self.rules, arity=2):
                        if (kids[0], kids[1]) == (left.label, right.label):
                            tree = PTree(key, prob)
                            tree.children = [left, right]
                            chart[j, i+j].append(tree)
            for element in chart[j, i+j]:
                for key, kids, prob in get_rules(self.rules, arity=1):
                    if kids[0] == element.label:
                        tree = PTree(key, prob)
                        tree.children = [element]
                        chart[j, i+j].append(tree)
        return chart[0, length-1]

    def POS_hook(self, word):
        """
        Given an unknown word, assign a class to it.

        You may assign this to a custom function that does POS tagging for a
        single word. By default, always assigns self.default_POS.
        """
        return self.default_POS

    def get_probability(self, sentence):
        """
        Return the probability of a given sentence.

        This sums up the probability of all parsed trees.
        """
        return sum(tree.probability for tree in self.parse(sentence))

if __name__ == '__main__':
    with open("testdata/testtrees.txt") as f:
        english = ContextFreeLanguage()
        english.learn_grammar(f)
        trees = english.parse("the woman eats with my man")
        print(trees[0])
        print(english.get_probability("the woman eats with my man"), "should be 0.004629629629629629")
