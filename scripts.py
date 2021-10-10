#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ## Say "Hello, World!" With Python

# In[ ]:


if __name__ == '__main__':
    print "Hello, World!"


# ## Python If-Else

# In[ ]:


#!/bin/python

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
if n%2 != 0:
    print "Weird"
else:
    if n>=2 and n<=5:   
        print "Not Weird"
    elif n>=6 and n<=20:    
        print "Weird"
    elif n>20:    
        print "Not Weird"


# ## Arithmetic Operators

# In[ ]:


if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print a+b
    print a-b
    print a*b


# ## Python: Division

# In[ ]:


from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
print a//b
print a/b


# ## Loops

# In[ ]:


if __name__ == '__main__':
    n = int(raw_input())
    for i in range(n):
        print i**2


# ## Write a function

# In[ ]:


def is_leap(year):
    leap = False
    
    if year%4 == 0:
        if year%100 == 0 and year%400 != 0:
            leap = False
        else:
            leap = True
            
    return leap


# ## Print Function

# In[ ]:


from __future__ import print_function

if __name__ == '__main__':
    n = int(raw_input())
print(*range(1,n+1), sep = '')


# # Data types

# ## List Comprehensions

# In[ ]:


if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    
    L = [[a,b,c] for a in xrange(x+1) for b in xrange(y+1) for c in xrange(z+1) if a+b+c != n]
        
    print L
    


# ## Find the Runner-Up Score!

# In[ ]:


if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    
    aux = sorted(arr, reverse = True)
    big = aux[0]
        
    for count in range(len(aux)):
        if aux[count] < big:
            big = aux[count] 
            break
        
    print big


# ## Nested Lists

# In[ ]:


if __name__ == '__main__':
    record = []
    for x in range(int(raw_input())):
        name = raw_input()
        score = float(raw_input())
        record.append([name,score])
        
    sort = sorted(record, key = lambda x: x[1])     
    mini = sort[0][1]

    for lst in sort:
        if lst[1] > mini:
            mini = lst[1] 
            break  
        
    names = []
    
    for lst in sort:
        if lst[1] == mini:
            names.append(lst[0])
            
    names_sort = sorted(names)  
                  
    for item in names_sort:
        print item        
         
 


# ## Finding the percentage

# In[ ]:


if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    
    out = sum(student_marks[query_name])/len(student_marks[query_name])
    
    print '%.2f' % out


# ## Lists

# In[ ]:


if __name__ == '__main__':
    N = int(raw_input())
    L = []
    for i in range(N):
        entry = raw_input().split()
        command = entry[0] 
        
        if command == "pop":
            L.pop()
        elif command == "append":
            L.append(int(entry[1]))
        elif command == "insert":
            L.insert(int(entry[1]),int(entry[2]))
        elif command == "remove":
            L.remove(int(entry[1]))
        elif command == "sort":
            L.sort()
        elif command == "reverse":
            L.reverse()
        elif command == "print":
            print L


# ## Tuples

# In[ ]:


if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    
    tupla = tuple(integer_list)
    
    print hash(tupla)


# # Strings

# ## sWAP cASE

# In[ ]:


def swap_case(s):
    
    return s.swapcase()


# ## String Split and Join

# In[ ]:


def split_and_join(line):
    line = line.split(" ")
    return  "-".join(line) 


# ## What's Your Name?

# In[ ]:


#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print ("Hello " +first + " " + last + "! You just delved into python." )


# ## Mutations 

# In[ ]:


def mutate_string(string, position, character):
    
    return string[:position] + character + string[position+1:]


# ## Find a string

# In[ ]:


def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string)+1):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count


# ## String Validators

# In[ ]:


if __name__ == '__main__':
    s = raw_input()
    
    print any(x.isalnum() for x in s)
    print any(x.isalpha() for x in s)
    print any(x.isdigit() for x in s)
    print any(x.islower() for x in s)
    print any(x.isupper() for x in s)
    


# ## Text Alignment

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
thickness = int(raw_input()) 
c = 'H'

for i in range(thickness):
    print (c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)

for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

for i in range((thickness+1)/2):
    print (c*thickness*5).center(thickness*6)    

for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)    

for i in range(thickness):
    print ((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)


# ## Text Wrap

# In[ ]:



def wrap(string, max_width):        
    return textwrap.fill(string, max_width)


# ## Designer Door Mat

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
N,M = map(int,raw_input().split())
for i in xrange(N/2):
    print '-' * ((M-i*6-2)/2) +  ('.|.' * (i*2 + 1)) + '-' * ((M-i*6-2)/2)
print '-' * ((M-7)/2) + 'WELCOME' + '-' * ((M-7)/2)
for i in xrange(N/2-1, -1, -1):
    print '-' * ((M-i*6-2)/2) +  ('.|.' * (i*2 + 1)) + '-' * ((M-i*6-2)/2)


# ## String Formatting

# In[ ]:


def print_formatted(number):
    space = len(bin(number))-2

    for i in xrange(1,number+1):
        print str(i).rjust(space), str(int(oct(i))).rjust(space), str(hex(i)[2:].upper()).rjust(space), str(bin(i)[2:]).rjust(space)


# ## Alphabet Rangoli

# In[ ]:


def print_rangoli(size):
    
    for i in range(size-1,-1,-1):
        for j in range(i):
            print(end="--")
        for j in range(size-1,i,-1):
            print(chr(j+97),end="-")
        for j in range(i,size):
            if j != size-1:
                print(chr(j+97),end="-")
            else:
                print(chr(j+97),end="")
        for j in range(2*i):
            print(end="-")
        print()
    for i in range(1,size):
        for j in range(i):
            print(end="--")
        for j in range(size-1,i,-1):
            print(chr(j+97),end="-")
        for j in range(i,size):
            if j != size-1:
                print(chr(j+97),end="-")
            else:
                print(chr(j+97),end="")
        for j in range(2*i):
            print(end="-")
        print()


# ## Capitalize!

# In[ ]:


# Complete the solve function below.
def solve(s):
    lst = s.split(' ')
    out = []
    for l in lst:
        out.append(l.capitalize())
    return ' '.join(map(str, out))        


# ## The Minion Game

# In[ ]:


def minion_game(string):
    vowels = set('AEIOU')
    s = k = 0
    for i in range(len(string)):
        if string[i] in vowels:
            k += len(string)-i
        else:
            s += len(string)-i

    if s > k:
        print('Stuart' , s)
    elif k > s:
        print('Kevin' , k)
    else:
        print('Draw')


# ## Merge the Tools!

# In[ ]:


def merge_the_tools(string, k):
    n = len(string)
    t = []
    for i in range(n//k):
       t.append(string[k*i:((i+1)*k)] )
    for ti in t:
        u = []
        for i  in ti:
            if i not in u:
                u.append(i)
        print(*u,sep = "")
        
    
       


# # Sets

# ## Introduction to Sets

# In[ ]:


def average(array):
    s = set(array)
    return sum(s)/len(s)


# ## No Idea!

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n,m = list(input().split())
arr = list(input().split())
A = set(input().split())
B = set(input().split())

happy = 0

for i in arr:
    if i in A:
        happy += 1
    elif i in B:
        happy -=1
        
print(happy)


# ## Symmetric Difference

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
N_list = map(int, input().split(' '))
M = int(input())
M_list = map(int, input().split(' '))

N_set = set(N_list)
M_set = set(M_list)

set1 = N_set.difference(M_set)
set2 = M_set.difference(N_set)

res = set1.union(set2)
res = sorted(res)

for i in res:
    print (i)


# ## Set .add()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
stamp = set()

for x in range(n):
    stamp.add(input())

print(len(stamp))


# ## Set .discard(), .remove() & .pop()

# In[ ]:


n = int(input())
s = set(map(int, input().split()))

for i in range(int(input())):
    command = input().split()
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.remove(int(command[1]))
    else:
        s.discard(int(command[1]))
print(sum(s))


# ## Set .union() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
eng = set(map(int, input().split()))
b = int(input())
fra = set(map(int, input().split()))

print(len(eng.union(fra)))


# ## Set .intersection() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
eng = set(map(int, input().split()))
b = int(input())
fra = set(map(int, input().split()))

print(len(eng.intersection(fra)))


# ## Set .difference() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
eng = set(map(int, input().split()))
b = int(input())
fra = set(map(int, input().split()))

print(len(eng.difference(fra)))


# ## Set .symmetric_difference() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
eng = set(map(int, input().split()))
b = int(input())
fra = set(map(int, input().split()))

print(len(eng.symmetric_difference(fra)))


# ## Set Mutations

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
na = int(input())
A = set(int(x) for x in input().split(' '))

for i in range(int(input())):
    op,leng = input().split(' ')
    b = set(int(x) for x in input().split(' '))
    
    if op == "update":
        A.update(b)
    elif op == "intersection_update":
        A.intersection_update(b)
    elif op == "difference_update":
        A.difference_update(b)
    elif op == "symmetric_difference_update":
        A.symmetric_difference_update(b)
        
print(sum(A))
    
        


# ## The Captain's Room

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
k = int(input())
rooms = list(int(x) for x in input().split(' '))
cap = set(rooms)

aux = sum(cap) * k - sum(rooms)

print(int(aux/(k-1)))


# ## Check Subset

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
for  i in range(int(input())):
    nA = int(input())
    A = set(input().split(' '))
    nB = int(input())    
    B = set(input().split(' '))
    print(A.issubset(B))


# ## Check Strict Superset

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(input().split(' '))
ans = True
nt = int(input())
for i in range(nt):
    B = set(input().split(' '))
    if(A > B) == False:
        ans = False
        break
print(ans)
 


# # Collections

# ## collections.Counter()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter

X = int(input())
shoe = Counter(map(int,input().split(' ')))
N = int(input())
gain = 0

for i in range(N):
    size,price = map(int,input().split(' '))
    if shoe[size]>0:
        shoe[size] -=1
        gain += price
        
print (gain)


# ## DefaultDict Tutorial

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

d = defaultdict(list)
n,m = map(int,input().split(' '))

for i in range(1,n+1):
    d[input()].append(str(i))
    
for i in range(m):
    w = input()
    if w in d:
        print(*d[w])
    else:
        print(-1)


# ## Collections.namedtuple()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple

N = int(input())
stud = namedtuple('stud',input().split())

print(sum(int(stud(*input().split()).MARKS) for _ in range(N))/N)
    


# ## Collections.OrderedDict()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict

N = int(input())
d = OrderedDict()

for _ in range(N):
    prod = input().split()
    item_name, net_price = ' '.join(prod[:-1]), int(prod[-1])
    if item_name in d:
        d[item_name] += net_price
    else:
        d[item_name] = net_price

for item in d:
    print(item, d[item])


# ## Word Order

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

d = defaultdict()

for _ in range(int(input())):
    word = input()
    if word in d:
        d[word] += 1
    else:
        d[word] = 1

print(len(d))
print(*d.values())


# ## Collections.deque()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque

d=deque()
N=int(input())

for i in range(N):
    A=list(input().split())
    if A[0]=='append':
        d.append(int(A[1]))
    elif A[0]=='appendleft':
        d.appendleft(int(A[1]))
    elif A[0]=='pop':
        d.pop()
    elif A[0]=='popleft':
        d.popleft()

print (*d)


# ## Company Logo

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

from collections import Counter

if __name__ == '__main__':
    s = input()
    count = Counter(char for char in s).most_common()
    count = sorted(sorted(count, key = lambda x : x[0]) , key = lambda x :x[1], reverse = True)
    
    for i in range(3):
        print(*count[i])


# ## Piling Up!

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque

for _ in range(int(input())):
    n = int(input())
    side = deque(map(int, input().split(' ')))
    res = 'Yes'
    
    if side[0]<side[-1]:
        pile = side.pop()
    else:
        pile = side.popleft()   
   
    while len(side)>0:
        if pile>=side[-1] and side[0]<=side[-1]:
            pile = side.pop()
        elif pile>=side[0]:
            pile = side.popleft()              
        else:
            res = 'No'
            break            
        
    print (res)


# # Date and Time

# ## Calendar Module

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar

MM, DD, YYYY = map(int, input().split())
day = calendar.weekday(YYYY, MM, DD)

print (calendar.day_name[day].upper())


# ## Time Delta

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys
import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    time1 = datetime.datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    time2 = datetime.datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    
    return (str(math.trunc(abs((time2-time1).total_seconds()))))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


# # Exceptions

# ## Exceptions

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
for _ in range(int(input())):
    a,b = input().split()
    try:
        print(int(a)//int(b))
    except(ZeroDivisionError, ValueError) as e:
        print("Error Code:", e)


# # Builts-in

# ## Zipped!

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
N,X = map(int,input().split())
sub = []

for _ in range(X):
    mark = list(map(float,input().strip().split()))
    sub.append(mark)
sub = zip(*sub)

print('\n'.join(str(sum(x)/X) for x in sub))


# ## Input()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
x,k = map(int, raw_input().split())

print(input() == k)


# ## Python Evaluation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
eval(input())


# # Python Functionals 

# ## Map and Lambda Function

# In[ ]:


cube = lambda x: x**3 

def fibonacci(n):
    fib = [0,1,1]
    if n==0:
        return ([])
    elif n==1:
        return (fib[:1])
    elif n==2:
        return (fib[:2])
    else:
        a = fib[1]
        b = fib[2]
        for _ in range(n-3):
            fib.append(a+b)
            a,b = b,a+b
    return(fib)  


# # Regex and Parsing

# ## Detect Floating Point Number

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

forma = "[\+\-]?\d*\.{1}\d+$"
n = int(input())

for _ in range(n):
    print (bool(re.match(forma , input())))


# ## Re.split()

# In[ ]:


regex_pattern = r"[,.]"


# ## Group(), Groups() & Groupdict()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

rep = re.search(r"([a-z0-9])\1" ,input())
if rep != None:
    print(rep.group(1))
else:
    print('-1')


# ## Re.findall() & Re.finditer()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

vow = '[AEIOUaeiou]'
cons = '[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]'
forma = r'(%s)(%s{2,})(?=%s)' %(cons, vow, cons)

string = re.finditer(forma,input())
res = list(map(lambda x: x.group(2),string))

if res:
    for s in res:
        print (s)
else:
    print('-1')


# ## Re.start() & Re.end()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

S = input()
k = input()

m = re.finditer(r'(?='+k+')',S)
flag = 0
if m:
    for el in m:
        print('(%i, %i)' % (el.start(), el.start()+len(k)-1))
        flag = 1
if flag == 0:
    print('(-1, -1)')


# ## Validating Roman Numerals

# In[ ]:


regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.


# ## Validating phone numbers

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

forma = r"^[7-9]{1}\d{9}\n?\r?$"

for _ in range(int(input())):
    if(re.match(forma,input())):
        print("YES")
    else:
        print("NO")


# ## Validating and Parsing Email Addresses

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import email.utils
import re
forma = r"^<([a-z][\w\.\-]*)@([a-z]+)\.([a-z]{1,3})>$"

for _ in range(int(input())):
    name, S = map(str,input().split())
    m = re.match(forma, S)
    if m:   
        print (email.utils.formataddr((name, S[1:-1])))


# ## Hex Color Code

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
forma1 = "\{(.+?)\}"
forma2 = "(#[a-fA-F0-9]{6}|#[a-fA-F0-9]{3})"

n = int(input())
S = ''.join(input() for _ in range(n))

res = re.findall(forma1,S)

for l in res:
    res2 = re.findall(forma2,l)
    for col in res2:
        print(col)


# ## Validating UID

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

for _ in range(int(input())):
    S = input()
    up = re.search(r"[A-Z].*[A-Z]",S)
    num = re.search(r"[0-9].*[0-9].*[0-9]",S)
    leng = re.search(r"^[0-9a-zA-Z]{10}$",S)
    rep = re.search(r"(.).*\1",S)
    
    if up and num and leng and not rep:
        print("Valid")
    else:
        print("Invalid")


# # XML

# ## XML 1 - Find the Score

# In[ ]:




get_attr_number = lambda node : len(node.attrib) + sum(get_attr_number(child) for child in node);
    


# ## XML2 - Find the Maximum Depth

# In[ ]:




maxdepth = 0
def depth(elem,level):
    global maxdepth 
    maxdepth = auxdepth(elem)

def auxdepth(elem):
    if elem:
        return 1 + max([auxdepth(child) for child in elem])
    else:
        return 0
    


# # Closures and Decorations

# ## Standardize Mobile Number Using Decorators

# In[ ]:


def wrapper(sort):
    def fun(args):
        for i in range(len(args)):
            args[i] = "+91 "+args[i][-10:-5]+" "+args[i][-5:]
        sort(args)
    return fun


# ## Decorators 2 - Name Directory

# In[ ]:


def person_lister(f):
    def inner(people):
        people = sorted(people, key = lambda x : int(x[2]))
        return [f(i) for i in people]
    return inner


# # Numpy

# ## Arrays

# In[ ]:


def arrays(arr):
   return numpy.array(arr[::-1], float)


# ## Shape and Reshape

# In[ ]:


import numpy as np

arr = list(map(int, input().split()))
print (np.reshape(arr,(3,3)))


# ## Transpose and Flatten

# In[ ]:


import numpy as np

N,M = map(int,input().split())
mat = np.array([list(map(int, input().split())) for i in range(N)])
print (np.transpose(mat))
print (mat.flatten())


# ## Concatenate

# In[ ]:


import numpy

N,M,P = map(int, input().split())
mat1 = numpy.array([list(map(int, input().split())) for i in range(N)])
mat2 = numpy.array([list(map(int, input().split())) for i in range(M)])

print(numpy.concatenate((mat1,mat2), axis = 0))



# ## Zeros and Ones

# In[ ]:


import numpy

N = list(map(int,input().split()))

print (numpy.zeros((N), dtype = int))
print (numpy.ones((N), dtype = int))


# ## Eye and Identity

# In[ ]:


import numpy

numpy.set_printoptions(legacy='1.13')
print (numpy.eye(*list(map(int, input().split()))))


# ## Array Mathematics

# In[ ]:


import numpy

N,M = map(int, input().split())
A = numpy.array([list(map(int, input().split())) for i in range(N)])
B = numpy.array([list(map(int, input().split())) for i in range(N)])
print(A+B,A-B,A*B,A//B,A%B,A**B,sep = '\n')


# ## Floor, Ceil and Rint

# In[ ]:


import numpy

A = numpy.array(list(map(float, input().split())))
numpy.set_printoptions(legacy = '1.13')

print(numpy.floor(A),numpy.ceil(A), numpy.rint(A), sep = '\n')


# ## Sum and Prod

# In[ ]:


import numpy

N,M = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for i in range(N)])
A = numpy.sum(arr, axis = 0)
print(numpy.prod(A))


# ## Min and Max

# In[ ]:


import numpy

N,M = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for i in range(N)]) 
print (numpy.max(numpy.min(arr, axis = 1)))


# ## Mean, Var, and Std

# In[ ]:


import numpy

N,M = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for i in range(N)])
print(numpy.mean(arr,axis = 1), numpy.var(arr,axis = 0), numpy.around(numpy.std(arr,axis = None), decimals=11),sep = '\n')


# ## Dot and Cross

# In[ ]:


import numpy

N = int(input())
A = numpy.array([list(map(int, input().split())) for i in range(N)])
B = numpy.array([list(map(int, input().split())) for i in range(N)])

print(numpy.dot(A,B))


# ## Inner and Outer

# In[ ]:


import numpy

A = numpy.array(list(map(int, input().split())) )
B = numpy.array(list(map(int, input().split())) )

print (numpy.inner(A, B))
print (numpy.outer(A, B))



# ## Polynomials

# In[ ]:


import numpy

coeff = numpy.array(list(map(float, input().split())))
val = float(input())
print (numpy.polyval(coeff, val))


# ## Linear Algebra

# In[ ]:


import numpy

N = int(input())
arr = numpy.array([list(map(float, input().split())) for i in range(N)])

print (numpy.round(numpy.linalg.det(arr),decimals=2))


# # Birthday Cake Candels

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    candles = Counter(candles)
    return candles[max(candles.keys())]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# # Kangaroo

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    return 'YES' if v1>v2 and (x1-x2)%(v1-v2)==0 else 'NO'            
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# # Strange advertising

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    people = 5
    like = people//2
    people = people//2*3
    
    for _ in range(n-1):
        like += people//2 
        people = people//2*3    
    
    return like

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# # Recursive digit sum

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    return auxSuperDigit(str(int(digitSum(n)))*k)

def auxSuperDigit(n):
    if len(n) <= 1:
        return n
    else:
        return auxSuperDigit(digitSum(n))

def digitSum(x):
    return str(sum((int(i)) for i in str(x)))
        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# # Insertion sort 1

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    a = arr[n-1]
    j = n-2
    while j>=0 and arr[j]>a:
        arr[j+1]=arr[j]
        j -=1
        print(*arr)
    arr[j+1] = a
    print(*arr)   
    
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# # Insertion sort 2

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for i in range(1,n):
        a = arr[i]
        j = i-1
        while j>=0 and arr[j]>a:
            arr[j+1]=arr[j]
            j -=1
        arr[j+1] = a
        print(*arr) 
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

