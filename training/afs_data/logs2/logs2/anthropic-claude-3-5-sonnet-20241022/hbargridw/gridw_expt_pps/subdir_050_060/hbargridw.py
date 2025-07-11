import collections
from typing import List, Tuple

Point = Tuple[int, int]


@traced
def extract_start(sentence: str) -> Tuple[int, int]:
    """Extract the starting point for navigation from a sentence.
    """
    ...


@traced
def extract_goal(sentence: str) -> Tuple[int, int]:
    """Extract the ending point for navigation from a sentence.
    """
    ...


@traced
def at_goal(state: Point, goal: Point) -> bool:
    """Decide if the state is a goal state.
    """
    ...


@traced
@traced
def possible_actions(state: Point) -> List[str]:
    """Return a list of actions possible at this state.

    >>> possible_actions((3, 2))
    Calling possible_actions((3, 2))...
    ...possible_actions returned ['north', 'south', 'west']
    ['north', 'south', 'west']

    >>> possible_actions((3, 2))
    Calling possible_actions((3, 2))...
    ...possible_actions returned ['north', 'south', 'west']
    ['north', 'south', 'west']

    >>> possible_actions((5, 4))
    Calling possible_actions((5, 4))...
    ...possible_actions returned ['north', 'south', 'east']
    ['north', 'south', 'east']

    >>> possible_actions((3, 0))
    Calling possible_actions((3, 0))...
    ...possible_actions returned ['north', 'east', 'west']
    ['north', 'east', 'west']

    >>> possible_actions((3, 6))
    Calling possible_actions((3, 6))...
    ...possible_actions returned ['north', 'south', 'west']
    ['north', 'south', 'west']

    >>> possible_actions((5, 2))
    Calling possible_actions((5, 2))...
    ...possible_actions returned ['north', 'south', 'east']
    ['north', 'south', 'east']

    >>> possible_actions((5, 5))
    Calling possible_actions((5, 5))...
    ...possible_actions returned ['north', 'south', 'east']
    ['north', 'south', 'east']

    >>> possible_actions((5, 7))
    Calling possible_actions((5, 7))...
    ...possible_actions returned ['south', 'east']
    ['south', 'east']

    >>> possible_actions((5, 6))
    Calling possible_actions((5, 6))...
    ...possible_actions returned ['north', 'south', 'east']
    ['north', 'south', 'east']

    """
    ...


@traced
@traced
def optimal_actions(state: Point, possible: List[str], goal: Point) -> List[str]:
    """Return the actions in the list of possible actions that are optimal.

    >>> optimal_actions((7, 1), ['north', 'south', 'west'], (2, 4))
    Calling optimal_actions((7, 1), ['north', 'south', 'west'], (2, 4))...
    ...optimal_actions returned ['west']
    ['west']

    >>> optimal_actions((2, 1), ['north', 'south', 'east', 'west'], (7, 2))
    Calling optimal_actions((2, 1), ['north', 'south', 'east', 'west'], (7, 2))...
    ...optimal_actions returned ['east']
    ['east']

    >>> optimal_actions((7, 1), ['north', 'south', 'west'], (0, 5))
    Calling optimal_actions((7, 1), ['north', 'south', 'west'], (0, 5))...
    ...optimal_actions returned ['west']
    ['west']

    >>> optimal_actions((5, 4), ['north', 'south', 'east'], (2, 7))
    Calling optimal_actions((5, 4), ['north', 'south', 'east'], (2, 7))...
    ...optimal_actions returned ['south']
    ['south']

    """
    ...


@traced
def do_action(state: Point, action: str) -> Point:
    """Perform an action.
    """
    ...


def hbargridw_77(input_str):
    """Navigate from point to point on a grid.

  >>> hbargridw_77('What directions should be followed to navigate from (1,2) to (6,5)?')
  Calling extract_start('What directions should be followed to navigate from (1,2) to (6,5)?')...
  ...extract_start returned (1, 2)
  Calling extract_goal('What directions should be followed to navigate from (1,2) to (6,5)?')...
  ...extract_goal returned (6, 5)
  Calling at_goal((1, 2), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((1, 2))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((1, 2), ['north', 'south', 'east', 'west'], (6, 5))...
  ...optimal_actions returned ['south', 'east']
  Calling do_action((1, 2), 'south')...
  ...do_action returned (1, 1)
  Calling at_goal((1, 1), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((1, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((1, 1), ['north', 'south', 'east', 'west'], (6, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((1, 1), 'east')...
  ...do_action returned (2, 1)
  Calling at_goal((2, 1), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((2, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((2, 1), ['north', 'south', 'east', 'west'], (6, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((2, 1), 'east')...
  ...do_action returned (3, 1)
  Calling at_goal((3, 1), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((3, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((3, 1), ['north', 'south', 'east', 'west'], (6, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((3, 1), 'east')...
  ...do_action returned (4, 1)
  Calling at_goal((4, 1), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((4, 1))...
  ...possible_actions returned ['south', 'east', 'west']
  Calling optimal_actions((4, 1), ['south', 'east', 'west'], (6, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((4, 1), 'east')...
  ...do_action returned (5, 1)
  Calling at_goal((5, 1), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((5, 1), ['north', 'south', 'east', 'west'], (6, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 1), 'north')...
  ...do_action returned (5, 2)
  Calling at_goal((5, 2), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 2))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 2), ['north', 'south', 'east'], (6, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 2), 'north')...
  ...do_action returned (5, 3)
  Calling at_goal((5, 3), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 3))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 3), ['north', 'south', 'east'], (6, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 3), 'north')...
  ...do_action returned (5, 4)
  Calling at_goal((5, 4), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 4))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 4), ['north', 'south', 'east'], (6, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 4), 'north')...
  ...do_action returned (5, 5)
  Calling at_goal((5, 5), (6, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 5))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 5), ['north', 'south', 'east'], (6, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((5, 5), 'east')...
  ...do_action returned (6, 5)
  Calling at_goal((6, 5), (6, 5))...
  ...at_goal returned True
  Final answer: south east east east east north north north north east
  ['south', 'east', 'east', 'east', 'east', 'north', 'north', 'north', 'north', 'east']

  >>> hbargridw_77('What directions should be followed to navigate from (3,3) to (7,5)?')
  Calling extract_start('What directions should be followed to navigate from (3,3) to (7,5)?')...
  ...extract_start returned (3, 3)
  Calling extract_goal('What directions should be followed to navigate from (3,3) to (7,5)?')...
  ...extract_goal returned (7, 5)
  Calling at_goal((3, 3), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((3, 3))...
  ...possible_actions returned ['north', 'south', 'west']
  Calling optimal_actions((3, 3), ['north', 'south', 'west'], (7, 5))...
  ...optimal_actions returned ['south']
  Calling do_action((3, 3), 'south')...
  ...do_action returned (3, 2)
  Calling at_goal((3, 2), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((3, 2))...
  ...possible_actions returned ['north', 'south', 'west']
  Calling optimal_actions((3, 2), ['north', 'south', 'west'], (7, 5))...
  ...optimal_actions returned ['south']
  Calling do_action((3, 2), 'south')...
  ...do_action returned (3, 1)
  Calling at_goal((3, 1), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((3, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((3, 1), ['north', 'south', 'east', 'west'], (7, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((3, 1), 'east')...
  ...do_action returned (4, 1)
  Calling at_goal((4, 1), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((4, 1))...
  ...possible_actions returned ['south', 'east', 'west']
  Calling optimal_actions((4, 1), ['south', 'east', 'west'], (7, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((4, 1), 'east')...
  ...do_action returned (5, 1)
  Calling at_goal((5, 1), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((5, 1), ['north', 'south', 'east', 'west'], (7, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 1), 'north')...
  ...do_action returned (5, 2)
  Calling at_goal((5, 2), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 2))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 2), ['north', 'south', 'east'], (7, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 2), 'north')...
  ...do_action returned (5, 3)
  Calling at_goal((5, 3), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 3))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 3), ['north', 'south', 'east'], (7, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 3), 'north')...
  ...do_action returned (5, 4)
  Calling at_goal((5, 4), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 4))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 4), ['north', 'south', 'east'], (7, 5))...
  ...optimal_actions returned ['north', 'east']
  Calling do_action((5, 4), 'north')...
  ...do_action returned (5, 5)
  Calling at_goal((5, 5), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((5, 5))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 5), ['north', 'south', 'east'], (7, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((5, 5), 'east')...
  ...do_action returned (6, 5)
  Calling at_goal((6, 5), (7, 5))...
  ...at_goal returned False
  Calling possible_actions((6, 5))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((6, 5), ['north', 'south', 'east', 'west'], (7, 5))...
  ...optimal_actions returned ['east']
  Calling do_action((6, 5), 'east')...
  ...do_action returned (7, 5)
  Calling at_goal((7, 5), (7, 5))...
  ...at_goal returned True
  Final answer: south south east east north north north north east east
  ['south', 'south', 'east', 'east', 'north', 'north', 'north', 'north', 'east', 'east']

  >>> hbargridw_77('What directions should be followed to navigate from (0,6) to (5,6)?')
  Calling extract_start('What directions should be followed to navigate from (0,6) to (5,6)?')...
  ...extract_start returned (0, 6)
  Calling extract_goal('What directions should be followed to navigate from (0,6) to (5,6)?')...
  ...extract_goal returned (5, 6)
  Calling at_goal((0, 6), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((0, 6))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((0, 6), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['south', 'east']
  Calling do_action((0, 6), 'south')...
  ...do_action returned (0, 5)
  Calling at_goal((0, 5), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((0, 5))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((0, 5), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['south', 'east']
  Calling do_action((0, 5), 'south')...
  ...do_action returned (0, 4)
  Calling at_goal((0, 4), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((0, 4))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((0, 4), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['south', 'east']
  Calling do_action((0, 4), 'south')...
  ...do_action returned (0, 3)
  Calling at_goal((0, 3), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((0, 3))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((0, 3), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['south', 'east']
  Calling do_action((0, 3), 'south')...
  ...do_action returned (0, 2)
  Calling at_goal((0, 2), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((0, 2))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((0, 2), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['south', 'east']
  Calling do_action((0, 2), 'south')...
  ...do_action returned (0, 1)
  Calling at_goal((0, 1), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((0, 1))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((0, 1), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['east']
  Calling do_action((0, 1), 'east')...
  ...do_action returned (1, 1)
  Calling at_goal((1, 1), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((1, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((1, 1), ['north', 'south', 'east', 'west'], (5, 6))...
  ...optimal_actions returned ['east']
  Calling do_action((1, 1), 'east')...
  ...do_action returned (2, 1)
  Calling at_goal((2, 1), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((2, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((2, 1), ['north', 'south', 'east', 'west'], (5, 6))...
  ...optimal_actions returned ['east']
  Calling do_action((2, 1), 'east')...
  ...do_action returned (3, 1)
  Calling at_goal((3, 1), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((3, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((3, 1), ['north', 'south', 'east', 'west'], (5, 6))...
  ...optimal_actions returned ['east']
  Calling do_action((3, 1), 'east')...
  ...do_action returned (4, 1)
  Calling at_goal((4, 1), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((4, 1))...
  ...possible_actions returned ['south', 'east', 'west']
  Calling optimal_actions((4, 1), ['south', 'east', 'west'], (5, 6))...
  ...optimal_actions returned ['east']
  Calling do_action((4, 1), 'east')...
  ...do_action returned (5, 1)
  Calling at_goal((5, 1), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((5, 1))...
  ...possible_actions returned ['north', 'south', 'east', 'west']
  Calling optimal_actions((5, 1), ['north', 'south', 'east', 'west'], (5, 6))...
  ...optimal_actions returned ['north']
  Calling do_action((5, 1), 'north')...
  ...do_action returned (5, 2)
  Calling at_goal((5, 2), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((5, 2))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 2), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['north']
  Calling do_action((5, 2), 'north')...
  ...do_action returned (5, 3)
  Calling at_goal((5, 3), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((5, 3))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 3), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['north']
  Calling do_action((5, 3), 'north')...
  ...do_action returned (5, 4)
  Calling at_goal((5, 4), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((5, 4))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 4), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['north']
  Calling do_action((5, 4), 'north')...
  ...do_action returned (5, 5)
  Calling at_goal((5, 5), (5, 6))...
  ...at_goal returned False
  Calling possible_actions((5, 5))...
  ...possible_actions returned ['north', 'south', 'east']
  Calling optimal_actions((5, 5), ['north', 'south', 'east'], (5, 6))...
  ...optimal_actions returned ['north']
  Calling do_action((5, 5), 'north')...
  ...do_action returned (5, 6)
  Calling at_goal((5, 6), (5, 6))...
  ...at_goal returned True
  Final answer: south south south south south east east east east east north north north north north
  ['south', 'south', 'south', 'south', 'south', 'east', 'east', 'east', 'east', 'east', 'north', 'north', 'north', 'north', 'north']

  """
    ...
