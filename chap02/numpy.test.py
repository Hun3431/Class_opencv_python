import numpy as np

# list와 numpy array의 차이 1
# 선언 조건 1. 자료형 :
# list는 각각의 원소의 자료형이 달라도 사용이 가능함
# numpy array는 모든 원소의 자료형이 같아야 함.
# 정수와 문자열이 모두 들어오게 되면 정수도 문자열로 전환됨.
# 선언 조건 2. 배열 원소 개수 : (2차원 이상의 배열에서 해당)
# list는 리스트 내부의 배열에서 원소 개수가 달라져도 사용 가능
# numpy array는 내부 배열의 원소 개수가 다르면 선언 불가능

a = [1, 2, 3, 'a', 'b', 'c']
print(a)    #[1, 2, 3, 'a', 'b', 'c']
b = np.array([1, 2, 3, 'a', 'b', 'c'])
print(b)    #['1' '2' '3' 'a' 'b' 'c']

a2 = [[1], [2, 3], [4, 5, 6]]
print(a2)   #[[1], [2, 3], [4, 5]]
# b2 = np.array([[1], [2, 3], [4, 5, 6]]) # error message

# list와 numpy array의 차이 2
# 연산 :
# list의 덧셈은 값을 더하는게 아니라 항목을 이어 붙이는 작업을 수행함
# - : 동일 / * 자연수 : 해당list를 자연수 배 작업을 수행함]
# numpy array의 덧셈은 array 내부의 값을 더하는 작업을 수행함
# 나머지 사칙연산도 동일하게 내부 값의 연산이 이루어짐

arra = [1, 2, 3]
arrb = [4, 5, 6]
print(arra)         #[1, 2, 3]
print(arrb)         #[4, 5, 6]
print(arra+arrb)    #[1, 2, 3, 4, 5, 6

npa = np.array([1, 2, 3])
npb = np.array([4, 5, 6])
print(npa)          #[1 2 3]
print(npb)          #[4 5 6]
print(npa+npb)      #[5 7 9]

# list와 numpy array의 차이 3
# 메서드 :
# 자료형 종류가 다르기 때문에 메소드의 종류가 다름

a2 = [1, 2, 3]
b2 = np.array([1, 2, 3])

# 배열 길이
print(len(a2))      #3
print(b2.size)      #3

# 배열 원소 자료형
print(type(a2[0]))  #<class 'int'>
print(b2.dtype)     #int64

#
import time
# list와 numpy array의 차이 4
# 연산 속도 :
# numpy array가 연산 최적화가 더 잘되어있음.

a3 = list(range(10 ** 7))   # list에 0부터 9999999까지의 값을 a3에 list형태로 저장
start = time.time()
for i in range(10 ** 7):    # 각 원소를 두 배 시켜줌
    a3[i] *= 2
end = time.time()
print(end - start)          # 1.0341260433197021

b3 = np.array(range(10 ** 7))   # numpy array에 0부터 9999999까지의 값을 저장
start = time.time()
b3 *= 2                     # 각 원소를 두 배 시켜줌
end = time.time()
print(end - start)          # 0.009348869323730469
