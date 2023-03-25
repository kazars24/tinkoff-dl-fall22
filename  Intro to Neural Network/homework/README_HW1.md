{\rtf1\ansi\ansicpg1251\cocoartf2638
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # 01. PyTorch\
\
### Task 1 \
Calculate, using PyTorch, the sum of the elements of the range from 0 to 10000.\
\
### Task 2\
Solve optimization problem: find the minimum of the function f(x) = ||Ax^2 + Bx + C||^2, where\
 - x is vector of size 8\
 - A is identity matrix of size 8x8\
 - B is matrix of size 8x8, where each element is 0\
 - C is vector of size 8, where each element is -1\
\
Use PyTorch and autograd function. Relative error will be less than 1e-3\
\
Solution here is x, converted to the list(see submission.yaml).\
\
### Task 3\
Solve optimization problem: find the optimal parameters of the linear regression model, using PyTorch.\
\
        train_X = [[0, 0], [1, 0], [0, 1], [1, 1]],\
        \
        train_y = [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746]\
        text_X = [[0, -1], [-1, 0]]\
        \
User PyTorch. Relative error will be less than 1e-1\
        \
Solution here is test_y, calculated from test_X, converted to the list(see submission.yaml).\
}