import math

def input_location():
    ax, ay = map(int, input("a의 x, y 좌표를 입력하시오. ex) 1 3 : ").split())
    bx, by = map(int, input("b의 x, y 좌표를 입력하시오. ex) 1 3 : ").split())
    return [ax, ay, bx, by]

def location_radian(arr):
    return math.atan2(arr[3]-arr[1], arr[2]-arr[0])

def change_degree(rad):
    return (rad * 180) / math.pi

arr = input_location()
rad = location_radian(arr)
deg = change_degree(rad)

print(arr)
print(rad)
print(deg, '°')
 