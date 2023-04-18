num1 = 3
num2 = 4

while num1 > 0:
    print(num1)
    num1 -= 2
    if num1 == 2: break
else:
    print('else')
# 3
# 1
# else

while num2 > 0:
    print(num2)
    num2 -= 2
    if num2 == 2: break
else:
    print('else')
# 4

print("out")
# out

''' while-else 문에서 else는 while문에서 조건에 충족하지 못해 나가질 때 실행하는 코드이며
    break로 while문 중간에 나가지는 경우에는 else문이 출력되지 않는다. '''