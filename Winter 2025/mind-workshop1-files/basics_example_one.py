def sum_of_numbers(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

list_from_1to5 = [1,2,3,4,5]
print(sum_of_numbers(list_from_1to5))
