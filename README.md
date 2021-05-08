
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
	A[start], A[i-1] = A[i-1], A[start]
	return i - 1
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().UnixNano())
	random := rand.Int()%(end-start+1) + start
	A[random], A[start] = A[start], A[random]
	return partition(A, start, end)
}
```

### 2. Heap Sort

```go
func heapSort(a []int) {
	heapSize := len(a)
	buildMaxHeap(a, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		a[0], a[i] = a[i], a[0] //堆顶(最大值)交换到末尾,堆顶元素和堆底元素交换 
		heapSize--              //把剩余待排序元素整理成堆
		maxHeapify(a, 0, heapSize)
	}
}
func buildMaxHeap(a []int, heapSize int) { // O(n)
	for i := heapSize / 2; i >= 0; i-- { // heapSize / 2后面都是叶子节点，不需要向下调整
		maxHeapify(a, i, heapSize)
	}
}
func maxHeapify(a []int, i, heapSize int) { // O(nlogn) 大根堆，如果堆顶节点小于叶子，向下调整 
	l, r, largest := i*2+1, i*2+2, i
	if l < heapSize && a[largest] < a[l] { //左儿子存在且大于a[largest]
		largest = l
	}
	if r < heapSize && a[largest] < a[r] { //右儿子存在且大于a[largest]
		largest = r
	}
	if largest != i {				
		a[largest], a[i] = a[i], a[largest] //堆顶调整为最大值
		maxHeapify(a, largest, heapSize)    //递归处理
	}
}
```


### 3. Merge Sort

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

### 4. Insertion Sort


```go
func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		tmp, j := A[i], i
		for j > 0 && tmp < A[j-1] {
			A[j] = A[j-1]
			j--
		}
		A[j] = tmp
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
	for i := 0; i < x; i++ { //x -> n-1 (x后面已完成)
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
