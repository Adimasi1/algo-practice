# Program that implements various sorting algorithms and data structures.
# Includes performance testing and analysis for some algorithms.

# Import necessary libraries
import random 
import math 
import time 
import pandas as pd # For data analysis and DataFrame creation
import matplotlib.pyplot as plt # For plotting graphs

# ==============================================================================
# 1) Insertion Sort
# ==============================================================================

def insertion_sort(A: list) -> list:
    """
    Implements the Insertion Sort algorithm. 
    Sorts array A in ascending order.
    Complexity: O(n^2) in the worst case (reverse sorted array), O(n) in the best case (already sorted array).
    """
    # Iterate from 1 up to len(A)-1. A[j] is the key to be inserted into the sorted sublist A[0..j-1]
    for j in range(1, len(A)):
        key = A[j] # The element to be inserted
        i = j - 1 # Index of the last element in the already sorted sublist (A[0..j-1])
        
        # Shift elements greater than key one position to the right
        while(i >= 0 and A[i] > key):
            A[i+1] = A[i]
            i -= 1
        
        # Insert the key into its correct position
        A[i+1] = key
    return A

def inversed_insertion_sort(A: list) -> list:
    """
    Implements Insertion Sort in descending order.
    """
    for j in range (1, len(A)):
        key = A[j]
        i = j - 1
        # Comparison is A[i] < key for descending order
        while(i >= 0 and A[i] < key):
            A[i+1] = A[i]
            i -= 1
        A[i+1] = key
    return A

# --- Basic Test for Insertion Sort ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
print("--- Insertion Sort Test ---")
print("Original Array:", A_base)
# Use a copy to avoid modifying the original array
A_sorted = insertion_sort(A_base.copy()) 
print("Sorted Array (Ascending):", A_sorted)
A_inverted = inversed_insertion_sort(A_base.copy())
print("Sorted Array (Descending):", A_inverted)
print("---------------------------\n")

# --- Performance Analysis for Insertion Sort ---
results_insertion = []
# Test on arrays of increasing lengths (from 1 to 1000 elements)
for n in range(1, 1001): 
    # Create an array of length 'n' with random numbers between 0 and 10
    tempArray = [random.randint(0, 10) for _ in range(n)]
    
    startInsertion = time.time()
    insertion_sort(tempArray)
    endInsertion = time.time()
    
    # Save the number of elements and execution time
    results_insertion.append({"num_elements": len(tempArray), "time": endInsertion - startInsertion})

# Create a Pandas DataFrame with the results
df_insertion = pd.DataFrame(results_insertion)
print("Insertion Sort Performance DataFrame:\n", df_insertion.head())

# Plot the results
plt.figure(figsize=(10, 6))
# Data is in columns 0 ('num_elements') and 1 ('time') of the DataFrame
plt.plot(df_insertion[df_insertion.columns[0]], df_insertion[df_insertion.columns[1]], marker="o", markersize=2)
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time (s)")
plt.title("Insertion Sort Performance")
plt.grid(True)
plt.show()


# ==============================================================================
# 2) Selection Sort
# ==============================================================================

def selection_sort(A: list):
    """
    Implements the Selection Sort algorithm. 
    Finds the minimum element in the unsorted sublist and swaps it 
    with the leftmost element.
    Complexity: O(n^2) in all cases (worst, average, best).
    """
    l = len(A)
    # Iterate over all elements of the array
    for i in range(l):
        # Assume the current element is the minimum
        min_index = i
        
        # Search for the smallest element in the rest of the array (A[i+1] to A[l-1])
        for j in range(i+1, l):
            if A[j] < A[min_index]:
                min_index = j
                
        # Swap the smallest element found with the current element (A[i])
        A[i], A[min_index] = A[min_index], A[i]

# --- Basic Test for Selection Sort ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
print("--- Selection Sort Test ---")
print("Original Array:", A_base)
# selection_sort modifies the array in-place, so pass a copy
A_selection_sorted = A_base.copy()
selection_sort(A_selection_sorted)
print("Sorted Array:", A_selection_sorted)
print("---------------------------\n")


# ==============================================================================
# 3) Merge Sort
# ==============================================================================

def merge(A: list, p: int, q: int, r: int):
    """
    Merges two sorted sublists A[p..q] and A[q+1..r] into a single sorted sublist.
    """
    # Create sublists L (left) and R (right)
    L = A[p:q+1]
    R = A[q+1:r+1]
    
    i = j = 0 # Indices for L and R
    k = p     # Index for the main array A
    
    # Compare and merge elements from L and R into A
    while i < len(L) and j < len(R) :
        if(L[i] <= R[j]):
            A[k] = L[i]
            i+=1
        else:
            A[k] = R[j]
            j+=1
        k+=1
        
    # Copy any remaining elements of L (if any)
    while i < len(L):
        A[k] = L[i]
        i+=1
        k+=1
        
    # Copy any remaining elements of R (if any)
    while j < len(R):
        A[k] = R[j]
        j+=1
        k+=1

def merge_sort(A: list, p: int, r: int):
    """
    Implements the Merge Sort algorithm (Divide and Conquer).
    Recursively sorts the array A[p..r].
    Complexity: O(n log n) in all cases.
    """
    if p < r:
        # Find the middle point
        q = math.floor((p+r)/2)
        
        # Sort the left half
        merge_sort(A, p, q)
        
        # Sort the right half
        merge_sort(A, q+1, r)
        
        # Merge the two sorted halves
        merge(A, p, q, r)

# --- Basic Test for Merge Sort ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
print("--- Merge Sort Test ---")
print("Original Array:", A_base)
A_merge_sorted = A_base.copy()
# Call merge_sort on the entire array
merge_sort(A_merge_sorted, 0, len(A_merge_sorted) - 1)
print("Sorted Array:", A_merge_sorted)
print("---------------------------\n")

# --- Performance Analysis for Merge Sort ---
results_merge = []
# Test on arrays of increasing lengths (from 1 to 100000 elements, step 1000 for speed)
for n in range(1, 100001, 1000): 
    # Create an array of length 'n' with random numbers between 0 and 10
    tempArray = [random.randint(0, 10) for _ in range(n)]
    
    startMerge = time.time()
    merge_sort(tempArray, 0, len(tempArray) -1)
    endMerge = time.time()
    
    results_merge.append({"num_elements": len(tempArray), "time": endMerge - startMerge})

df_merge = pd.DataFrame(results_merge)
print("Merge Sort Performance DataFrame:\n", df_merge.head())
print("Total execution time for Merge Sort tests:", df_merge["time"].sum())

# Plot the results
plt.figure(figsize=(10, 6))
# Data is in columns 0 ('num_elements') and 1 ('time') of the DataFrame
plt.plot(df_merge[df_merge.columns[0]], df_merge[df_merge.columns[1]], marker="o", markersize=3)
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time (s)")
plt.title("Merge Sort Performance")
plt.grid(True)
plt.show()


# ==============================================================================
# 4) Heap Sort
# ==============================================================================

def heap_left(i: int) -> int:
    """Returns the index of the left child of the node at index i (in an array-based heap)."""
    return 2*i + 1

def heap_right(i: int) -> int:
    """Returns the index of the right child of the node at index i (in an array-based heap)."""
    return 2*i + 2

def max_heapify(A: list, i: int, heap_size: int):
    """
    Maintains the max-heap property.
    Assuming the trees rooted at heap_left(i) and heap_right(i) are max-heaps, 
    max_heapify(A, i) moves the element A[i] down until the max-heap property is satisfied.
    """
    largest = i
    l_index = heap_left(i)
    r_index = heap_right(i)
    
    # Check if the left child is larger than the current root
    if l_index < heap_size and A[l_index] > A[largest]:
        largest = l_index
        
    # Check if the right child is larger than the largest so far
    if r_index < heap_size and A[r_index] > A[largest]:
        largest = r_index
        
    # If the largest is not the current root, swap and call recursively
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        max_heapify(A, largest, heap_size)

def build_max_heap(A: list):
    """
    Builds a max-heap from an unsorted array.
    Complexity: O(n).
    """
    heap_size = len(A)
    # Iterate from the first non-leaf node up to the root (len(A)//2 - 1)
    for i in range(len(A)//2 - 1, -1, -1):
        max_heapify(A, i, heap_size)

def heap_sort(A: list):
    """
    Implements the Heap Sort algorithm.
    Complexity: O(n log n) in all cases.
    """
    # 1. Build a max-heap from the array
    build_max_heap(A)
    heap_size = len(A)
    
    # 2. Extract the maximum (root) and move it to the end of the array, 
    # then restore the max-heap property on the remainder.
    for i in range(len(A)-1, 0, -1):
        # Swap the root (maximum element) with the last element of the current heap
        A[0], A[i] = A[i], A[0]
        
        # Decrease the heap size (the element is now sorted in A[i])
        heap_size -= 1
        
        # Restore the max-heap property on the new root
        max_heapify(A, 0, heap_size)

# --- Basic Test for Heap Sort ---
A_base = [27, 17, 3, 16, 13, 10, 1, 5, 7, 12, 4, 8, 9, 0]
print("--- Heap Sort Test ---")
print("Original Array:", A_base)
# heap_sort modifies the array in-place
heap_sort(A_base)
print("Sorted Array:", A_base)
print("---------------------------\n")


# ==============================================================================
# 5) Quick Sort
# ==============================================================================

def partition(A: list, p: int, r: int) -> int:
    """
    Chooses a pivot (A[r]) and partitions the array A[p..r] into two sublists:
    elements <= pivot and elements > pivot. Returns the final index of the pivot.
    """
    x = A[r] # Choose the last element as the pivot
    i = p - 1 # Index of the smallest element
    
    # Iterate from p to r-1
    for j in range(p, r):
        # If the current element is less than or equal to the pivot
        if A[j] <= x:
            i = i + 1
            # Swap A[i] and A[j]
            A[i], A[j] = A[j], A[i]
            
    # Swap the pivot (A[r]) with the element at position i+1
    i += 1 # New pivot index
    A[i], A[r] = A[r], A[i]
    
    return i # Return the final index of the pivot

def quicksort(A: list, p: int, r: int):
    """
    Implements the Quick Sort algorithm (Divide and Conquer).
    Recursively sorts the array A[p..r].
    Complexity: O(n log n) on average, O(n^2) in the worst case.
    """
    if p < r:
        # Partition the array and get the pivot index
        q = partition(A, p, r)
        
        # Recursively sort the left sublist (elements <= pivot)
        quicksort(A, p, q-1) 
        
        # Recursively sort the right sublist (elements > pivot)
        quicksort(A, q+1, r)

# --- Basic Test for Quick Sort ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
print("--- Quick Sort Test ---")
print("Original Array:", A_base)
A_quicksort_sorted = A_base.copy()
quicksort(A_quicksort_sorted, 0, len(A_quicksort_sorted) - 1)
print("Sorted Array:", A_quicksort_sorted)
print("---------------------------\n")

# --- Performance Analysis for Quick Sort ---
results_quicksort = []
# Test on arrays of increasing lengths (from 1 to 100000 elements, step 500)
for n in range(1, 100001, 500): 
    tempArray = [random.randint(0, 10) for _ in range(n)]
    
    startquicksort = time.time()
    quicksort(tempArray, 0, len(tempArray) -1 )
    endquicksort = time.time()
    
    results_quicksort.append({"num_elements": len(tempArray), "time": endquicksort - startquicksort})

df_quicksort = pd.DataFrame(results_quicksort)
print("Quick Sort Performance DataFrame:\n", df_quicksort.head())

# Plot the results
plt.figure(figsize=(10, 6))
# Data is in columns 0 ('num_elements') and 1 ('time') of the DataFrame
plt.plot(df_quicksort[df_quicksort.columns[0]], df_quicksort[df_quicksort.columns[1]], marker="o", markersize=3)
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time (s)")
plt.title("Quicksort Performance")
plt.grid(True)
plt.show()


# ==============================================================================
# 6) Counting Sort
# ==============================================================================

def counting_sort(A: list[int], k: int) -> list[int]:
    """
    Implements the Counting Sort algorithm. 
    Suitable for sorting integers within a limited range [0, k].
    Complexity: O(n + k).
    Parameters:
    - A: List of integers to be sorted.
    - k: The maximum value in array A.
    - B (Output Array): Output array for the sorted result.
    - C (Counting Array): Temporary array for counting.
    """
    
    # Array C for counting occurrences, initialized to 0. Size k+1 to include k.
    C = [0] * (k + 1)
    lenA = len(A)
    
    # Array B for the result, initialized to 0. Size len(A).
    B = [0] * lenA 
    
    # 1. Count the occurrences of each element in A and store them in C
    for j in range(lenA):
        C[A[j]] += 1
        
    # 2. Calculate cumulative sums in C. 
    # C[i] will now contain the number of elements less than or equal to i.
    for i in range(1, k+1):
        C[i] = C[i] + C[i-1]
        
    # 3. Place each element into its correct position in array B
    # Iterate backwards to maintain stability (equal elements preserve original order)
    for j in range(lenA -1, -1, -1):
        # The correct position is C[A[j]] - 1
        B[C[A[j]]-1] = A[j]
        # Decrement the count for this element
        C[A[j]] -= 1
        
    return B

# --- Basic Test for Counting Sort ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
k_max = max(A_base)
print("--- Counting Sort Test ---")
print("Original Array:", A_base)
# The function returns a new sorted array
A_counting_sorted = counting_sort(A_base, k_max)
print("Sorted Array:", A_counting_sorted)
print("---------------------------\n")


# ==============================================================================
# 7) Selection Algorithms
# ==============================================================================

def simple_selection(A: list) -> int:
    """
    Finds the maximum element in an unsorted array.
    Complexity: O(n).
    """
    if not A:
        return None # Handle empty list case
        
    maxA = A[0] # Initialize with the first element
    lenA = len(A)
    
    # Iterate to find the maximum
    for i in range(1, lenA):
        if(A[i] > maxA): 
            maxA = A[i]
            
    return maxA

# --- Basic Test for Simple Selection (Finding Max) ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
print("--- Simple Selection Test ---")
print("Array:", A_base)
max_result = simple_selection(A_base)
print("The maximum element (simple_selection) is:", max_result)
print("---------------------------\n")


def randomized_partition(A: list[int], p: int, r: int) -> int:
    """
    Partitions the array A[p..r] around a randomly chosen pivot.
    It is the core of Quickselect (Randomized Select).
    """
    # 1. Choose a random index z between p and r
    z = random.randint(p, r)
    
    # 2. Swap the random element A[z] with the last element A[r] (the pivot)
    A[z], A[r] = A[r], A[z]
    pivot = A[r]
    
    i = p - 1
    # 3. Partition (as in Quick Sort)
    for j in range(p, r):
        if A[j] <= pivot:
            i += 1
            A[i], A[j] = A[j], A[i]
            
    # 4. Put the pivot in its final position
    # Swap A[i+1] with A[r] (the original pivot)
    A[i+1], A[r] = A[r], A[i+1]
    
    return i + 1 # Return the final index of the pivot

def randomized_select(A: list[int], p: int, r: int, i: int) -> int:
    """
    Finds the i-th smallest element in A[p..r] (where i=1 is the smallest).
    Complexity: O(n) on average (expected time).
    Parameters:
    - A: List of integers.
    - p: Start index (inclusive).
    - r: End index (inclusive).
    - i: The position of the element to search for (1-based, so i=1 is the minimum).
    """
    if p == r:
        return A[p] # Base case: array with a single element
        
    # Partition the array
    q = randomized_partition(A, p, r) 
    
    # k is the 1-based position of the pivot A[q] within A[p..r]
    k = q - p + 1
    
    if i == k:
        # The pivot is the element we are looking for
        return A[q]
    elif (i < k):
        # The element is in the left sublist
        return randomized_select(A, p, q-1, i)
    else:
        # The element is in the right sublist. 
        # Search for the (i-k)-th smallest element in the right sublist.
        return randomized_select(A, q+1, r, i-k)

# --- Basic Test for Randomized Select ---
A_base = [13, 2, 4, 5, 10, 22, 3, 1, 0]
# Randomized Select modifies the array in-place during execution, so use a copy.
A_temp = A_base.copy() 
# Example: Search for the 4th smallest element (i=4)
# (Sorted elements are: 0, 1, 2, 3, 4, 5, 10, 13, 22. The 4th is 3)
select_result = randomized_select(A_temp, 0, len(A_temp)-1, 4)
print("--- Randomized Select Test ---")
print("Original Array:", A_base)
print("The result of randomized_select (4th smallest element) is:", select_result)
print("---------------------------\n")


# ==============================================================================
# 8) Data Structure: Stack
# ==============================================================================

class stack:
    """
    Implements a Stack data structure using a Python list.
    The Stack follows the LIFO (Last-In, First-Out) principle.
    It can have an optional maximum size.
    """
    def __init__(self, size=None):
        """
        Initializes a new Stack.
        :param size: The maximum size of the Stack (None for unlimited).
        """
        self.size = size
        self.stack = [] # The list used to store elements
        
    def isEmpty(self) -> bool:
        """Checks if the Stack is empty."""
        return len(self.stack) == 0

    def isFull(self) -> bool:
        """Checks if the Stack is full (only if a size was specified)."""
        return (self.size is not None) and (len(self.stack) >= self.size)

    def getStack(self):
        """Returns the internal list representing the Stack."""
        return self.stack

    def push(self, var) -> bool:
        """
        Adds an element to the top of the Stack ('push' operation).
        :param var: The element to add.
        :return: True if the addition was successful, False if the Stack is full.
        """
        if self.isFull():
            return False
        self.stack.append(var) # Appends to the end of the list
        return True

    def pop(self):
        """
        Removes and returns the element at the top of the Stack ('pop' operation).
        :return: The removed element, or None if the Stack is empty.
        """
        if self.isEmpty():
            return None
        return self.stack.pop() # Removes and returns the last element of the list

# --- Basic Test for the Stack class ---
print("--- Stack Test ---")
my_stack = stack(size=3) # Stack with a maximum size of 3
print(f"Is Stack empty? {my_stack.isEmpty()}") # Output: True

my_stack.push(10)
my_stack.push(20)
print(f"Push 10, 20. Stack: {my_stack.getStack()}") # Output: [10, 20]

my_stack.push(30)
print(f"Push 30. Stack: {my_stack.getStack()}") # Output: [10, 20, 30]
print(f"Is Stack full? {my_stack.isFull()}") # Output: True

print(f"Push 40 (fails): {my_stack.push(40)}") # Output: False

item = my_stack.pop()
print(f"Pop: {item}. Stack: {my_stack.getStack()}") # Output: 30. Stack: [10, 20]

item = my_stack.pop()
item = my_stack.pop()
print(f"Pop twice. Stack: {my_stack.getStack()}") # Output: [].
print(f"Pop on empty stack: {my_stack.pop}")
