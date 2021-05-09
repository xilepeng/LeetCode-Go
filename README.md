## Sorting

### 1. Quick Sort

```go
func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}
func partition(A []int, start, end int) int {
	piv, i := A[start], start+1
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[i-1], A[start] = A[start], A[i-1]
	return i - 1
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().Unix())
	random := start + rand.Int()%(end-start+1)
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```



### 2. Heap Sort

```go
func heap_sort(Arr []int, heap_size int) {
	build_maxheap(Arr, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		Arr[0], Arr[i] = Arr[i], Arr[0]
		heap_size--
		max_heapify(Arr, 0, heap_size)
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- {
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) {
	l, r, largest := i<<1+1, i<<1+2, i
	for l < heap_size && A[largest] < A[l] {
		largest = l
	}
	for r < heap_size && A[largest] < A[r] {
		largest = r
	}
	if largest != i {
		A[largest], A[i] = A[i], A[largest]
		max_heapify(A, largest, heap_size)
	}
}
```


### 3. Merge Sort

```go

func sortArray(nums []int) []int {
	merge_sort(nums, 0, len(nums)-1)
	return nums
}
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	// copy(A[start:end+1], Arr)
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
```

```go
func sortArray(nums []int) []int {
	n := len(nums)
	Arr := make([]int, n)
	merge_sort(nums, Arr, 0, n-1)
	return nums
}
func merge_sort(A, Arr []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		merge_sort(A, Arr, start, mid)
		merge_sort(A, Arr, mid+1, end)
		merge(A, Arr, start, mid, end)
	}
}
func merge(A, Arr []int, start, mid, end int) {
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] < A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	// copy(A[start:end+1], Arr)
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
```

```go
func sortArray(nums []int) []int {
	n := len(nums)
	temp := make([]int, n)
	mergeSort(nums, temp, 0, n-1)
	return nums
}
func mergeSort(A, temp []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		mergeSort(A, temp, start, mid)
		mergeSort(A, temp, mid+1, end)
		merge(A, temp, start, mid, end)
	}
}
func merge(A, temp []int, start, mid, end int) {
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if A[i] <= A[j] {
			temp[k] = A[i]
			i++
		} else {
			temp[k] = A[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		temp[k] = A[i]
		k++
	}
	for ; j <= end; j++ {
		temp[k] = A[j]
		k++
	}
	copy(A[start:end+1], temp)
}
```




### 4. Insertion Sort


```go
func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] {
			A[j] = A[j-1]
			j--
		}
		A[j] = temp
	}
}
```

### 5. Bubble Sort

```go
func bubble_sort(A []int, n int) {
	for k := 0; k < n; k++ {
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i]
			}
		}
	}
}
```


### 6. Selection Sort


```go
func selection_sort(A []int, n int) {
	for i := 0; i < n-1; i++ {
		min := i
		for j := i + 1; j < n; j++ {
			if A[j] < A[min] {
				min = j
			}
		}
		A[i], A[min] = A[min], A[i]
	}
}
```