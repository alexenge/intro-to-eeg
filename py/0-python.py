# %% [markdown]
# # 0. Intro to Python
#
# {{ badge }}
# <br>
#
# In this course, we will us the Python programming language for analyzing EEG data.
# Python as an open source programming language that is used in many areas of science and engineering.
# It is very similar to the [R language](https://www.r-project.org) that you might know from your statistics classes, but slightly more general and more widely used outside of psychology.
# This first section of the course will provide a very brief introduction into the Python language.
#
# In this section we will:
# - Use Python as a calculator
# - Define variables
# - Meet different data types
# - Run loops
# - Run functions
# - Load modules
#
# %% [markdown]
# ## 0.1 Calculations
#
# Like most programming languages, Python can perform basic arithmetic operations like addition, subtraction, multiplication, and division.
#
# %%
3 + 3

# %%
10 - 7

# %%
7 * 8

# %%
10 / 4

# %%
10 // 4

# %%
2 ** 4

# %%
16 ** 0.5

# %% [markdown]
#
# Using the hash sign (``#``), we can add comments to our code to add notes for ourselves or others.
#
# %%
# Comments (starting with '#') are ignored by Python

# %% [markdown]
#
# **Variables** are used to store values for later use.
# They are defined by assigning a value (right hand side) to a custom variable name (left hand side), with an equals sign (``=``) in between.
# We can print the value of a variable by simply typing its name (or, alternatively, by using the ``print()`` function).
#
# %% [markdown]
# ## 0.2 Variables
#
# %%
a = 3
a

# %%
a = 3
b = 4
c = a + b
c

# %%
a = 3
a = 5
a

# %%
a = 3
b = a
a = 5
b

# %% [markdown]
# ## 0.3 Data types
#
# So far, we have only used single numbers, but Python also supports many other types of data.
# An important distinction is between **scalars** (single values, such as a single number) and **iterables** (collections of values, such as a list of numbers).
#
# %% [markdown]
# ### 0.3.1 Scalars
#
# %% [markdown]
# An **integer** (``int``) is a whole number (positive or negative).
#
# %%
my_var = 3
type(my_var)

# %% [markdown]
#
# A **float** (``float``) is a number with a decimal point.
#
# %%
my_var = 3.0
type(my_var)

# %% [markdown]
#
# A **string** (``str``) is a sequence of characters (letters, numbers, symbols).
#
# %%
my_var = 'Hello world'
type(my_var)

# %% [markdown]
#
# A **Boolean** (``bool``) is one of the two logical values ``True`` or ``False``.
#
# %%
my_var = True
type(my_var)

# %% [markdown]
#
# **``None``** is a special value that represents the absence of a value.
#
# %%
my_var = None
type(my_var)

# %% [markdown]
# ### 0.3.2 Iterables
#
# %% [markdown]
#
# A **list** (``list``) is a collection of values and is defined by square brackets.
#
# %%
my_var = [1, 2, 3]
type(my_var)

# %% [markdown]
#
# Elements of a list can be accessed via their index (starting at ``0``, unlike R or MATLAB) in square brackets.
#
# %%
my_var[1]

# %% [markdown]
#
# This can also be used to change the value of an element.
# Note that, as shown here, lists can contain elements of different types (unlike vectors in R).
#
# %%
my_var[2] = 'Hooray!'
my_var

# %% [markdown]
#
# A **tuple** (``tuple``) works like a list, but cannot be changed after it has been created (it is "immutable").
# It is defined by round brackets.
#
# %%
my_var = (1, 2, 3)
type(my_var)

# %%
my_var[1]


# %% tags=["raises-exception"]
my_var[2] = 'Hooray!'

# %% [markdown]
#
# A **dictionary** (``dict``) is a collection of key-value pairs and is defined by curly brackets.
# The keys (usually some kind of labels) can then be used to access the corresponding values (usually some kinds of data).
#
# %%
my_var = {'Name': 'Berger', 'First_name': 'Hans', 'age': 53, 'married': True}
type(my_var)

# %%
my_var['First_name']

# %% [markdown]
# ## 0.4 Loops
#
# Loops are used to repeat a certain operation multiple times.
# The most common type of loop is a **for loop**, which iterates over the elements of an iterable (such as a list).
#
# %%
names = ['Alice', 'Bob', 'Chantoya']
for name in names:
    print('Hello ' + name)

# %% [markdown]
#
# Note that the operations inside the loop need to be indented by exactly four spaces.
# Python, unlike R, is very strict about indentation.
#
# It is often useful to store the results of a loop in a new list.
#
# %%
names = ['Alice', 'Bob', 'Chantoya']
greetings = []
for name in names:
    greetings.append('Hello ' + name)
greetings

# %% [markdown]
# ## 0.5 Functions
#
# Functions are pre-defined operations that can be applied to some input data.
# Functions are called by writing their name, followed by round brackets containing the input arguments.
#
# As an example, the ``sum()`` function calculates the sum of all elements in a list.
#
# %%
numbers = [1, 2, 3]
sum(numbers)

# %% [markdown]
#
# The ``len()`` function returns the length of an iterable.
#
# %%
len(numbers)

# %% [markdown]
#
# We can also define our own custom functions like this:
#
# %%


def square(x):
    """Multiplies a number (x) by itself."""
    return x ** 2


square(3)

# %% [markdown]
#
# Functions can have multiple input arguments.
# These can be specified by name (keyword arguments) or by position (positional arguments).
#
# %%


def power(base, exponent):
    """Exponentiates a number (base) to a given power (exponent)."""
    return base ** exponent


power(2, exponent=3)

# %% [markdown]
# ## 0.6 Modules
#
# Python only comes with a limited set of built-in functions like ``sum()`` and ``len()``.
# Many additional functions are provided by external **modules** (also called **packages**).
#
# To use these, we first need to install them (only once) and then import them (at the beginning of each script).
# As an example, we will install the ``numpy`` module for working with numerical data.
#
# %%
# %pip install numpy

# %%
# We can now import the module and use its functions:
#
# %%
import numpy

numbers = [1, 2, 3]
numpy.mean(numbers)

# %% [markdown]
#
# We can also import the module under a custom name (alias) to make it easier to use.
# This is very common for the ``numpy`` module, which is usually imported as ``np``.
#
# %%
import numpy as np

np.mean(numbers)

# %% [markdown]
#
# We can also import only specific functions from a module:
#
# %%
from numpy import mean

mean(numbers)

# %% [markdown]
# Numpy can also be used to create arrays, which are similar to lists but with two important differences:
# - Arrays can only contain elements of the same type (e.g., only floats)
# - Arrays can be multi-dimensional (e.g., a 2D matrix with two rows and three columns)

# %%
my_array = np.array([[1, 2, 3], [4, 5, 6]])
np.mean(my_array)

# %%
np.mean(my_array, axis=0)

# %%
np.mean(my_array, axis=1)
