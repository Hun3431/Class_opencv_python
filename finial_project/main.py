import numpy as np
import cv2
import copy

image_count = 310
width = 120
height = 150
size = width * height

image_files = []

for i in range(image_count):
    image = cv2.imread(f"./face_img/train/train{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (width, height))
    image_files.append(image)

sum_image = np.zeros((height, width), dtype=np.float32)

for i in range(len(image_files)):
    sum_image += np.array(image_files[i], dtype=np.float32)

average = sum_image / image_count

# cv2.imshow("Average Image", average.astype(np.uint8))

difference_array = np.zeros((size, 1), dtype=np.uint8)

for i in range(len(image_files)):
    image = copy.deepcopy(image_files[i])
    image = image - average
    image = image.reshape(size, 1)
    difference_array = np.append(difference_array, image, axis=1)

difference_array = np.delete(difference_array, 0, axis=1)

# print(difference_array)

covariance_array = np.cov(difference_array.T)

# print(covariance_array)

eigen_value, eigen_vector = np.linalg.eig(covariance_array)

# print(eigen_value)
# print(eigen_vector)

index = eigen_value.argsort()[:: -1]
eigen_value_sort = eigen_value[index]
eigen_vector_sort = eigen_vector[:, index]

# print(eigen_value_sort)
# print(eigen_vector_sort)

sum = 0
rate = 0.95
select_index = 0

sum_eigen_value = eigen_value_sort.sum() * rate
for i in range(len(eigen_value_sort)):
    sum += eigen_value_sort[i]

    if sum_eigen_value <= sum:
        select_index = i + 1
        break;

# print(select_index)

transform_matrix = np.zeros((size, 1))

for i in range(select_index):
    mul = (difference_array @ eigen_vector_sort[:, i]).reshape(size, 1)
    transform_matrix = np.append(transform_matrix, mul / np.linalg.norm(mul), axis=1)

transform_matrix = np.delete(transform_matrix, 0, axis=1)

# print(transform_matrix)

pca_array = np.zeros((select_index, 1))
for i in range(image_count):
    pca_array = np.append(pca_array, (transform_matrix.T @ difference_array[:, i]).reshape(select_index, 1), axis=1)

pca_array = np.delete(pca_array, 0, axis=1)

# print(pca_array)

test_image_count = 93
test_files = []

for i in range(test_image_count):
    image = cv2.imread(f"./face_img/test/test{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (width, height))
    test_files.append(np.array(image, np.float32))

print(test_files)

cv2.waitKey()