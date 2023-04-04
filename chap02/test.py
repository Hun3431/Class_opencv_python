a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print('a = ', a)                    # a =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print('a[:2] =>', a[:2])            # a[:2] => [0, 1]
print('a[4:-1] =>', a[4:-1])        # a[4:-1] => [4, 5, 6, 7, 8]
print('a[2::2] =>', a[2::2])        # a[2::2] => [2, 4, 6, 8]
print('a[::-1] =>', a[::-1])        # a[::-1] => [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print('a[1::-1] =>', a[1::-1])      # a[1::-1] => [1, 0]
print('a[7:1:-2] =>', a[7:1:-2])    # a[7:1:-2] => [7, 5, 3]
print('a[:-4:-1] =>', a[:-4:-1])    # a[:-4:-1] => [9, 8, 7]


words = ["솜씨좋은", "파이썬", "프로그래밍", "스터디", "python"]
words_copy = words
new_words = []

for word in words_copy:
    if len(word) > 3:
        new_words.append(word)
print(new_words)

words_copy = words

new_words = []
print(new_words)
new_words = [word for word in words_copy if len(word) > 3]

print(new_words)