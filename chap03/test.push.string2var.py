# 문자열에 변수 값 넣기

# Formatting string
name1 = "limdohun"
age1 = 24
message1 = "My name is %s. Age is %d." %(name1, age1)
print(message1)
# My name is limdohun. Age is 24.

# format 함수
arr = []
for i in range(5):
    arr.append("{0:02d}.jpg".format(i))
print(arr)
# ['00.jpg', '01.jpg', '02.jpg', '03.jpg', '04.jpg']

# f-string
name2 = "M-Queue"
age2 = 42
message2 = f"옆에서 떠드는 이친구는 {name2}입니다. 나이는 {age2}인거 같아요..."
print(message2)
# 옆에서 떠드는 이친구는 M-Queue입니다. 나이는 42인거 같아요...