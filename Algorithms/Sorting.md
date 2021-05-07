

## Sorting

[Quick Sort](https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/tutorial/)


```go
package main

import (
	"fmt"
	"math/rand"
	"time"
)

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

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	quick_sort(A, 0, n-1)
	for i := 0; i < n; i++ {
		fmt.Printf("%d ", A[i])
	}
}

```

[Heap Sort](https://www.hackerearth.com/practice/algorithms/sorting/heap-sort/tutorial/)

```go
package main

import (
	"fmt"
)

func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heap_size--
		max_heapify(A, 0, heap_size)
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

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	heap_sort(A)
	fmt.Println(A)
}

```


[Merge Sort](https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/tutorial/)

```go
package main

import (
	"fmt"
)

func merge_sort(A []int, start, end int) {
	if start < end {
		mid := (start + end) >> 1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	tmp := []int{}
	i, j := start, mid+1
	for i <= mid || j <= end {
		if i > mid || j <= end && A[j] < A[i] {
			tmp = append(tmp, A[j])
			j++
		} else {
			tmp = append(tmp, A[i])
			i++
		}
	}
	copy(A[start:end+1], tmp)
}

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	merge_sort(A, 0, len(A)-1)
	fmt.Println(A)
}

```


[Bubble Sort](https://www.hackerearth.com/practice/algorithms/sorting/bubble-sort/tutorial/)

```go
package main

import (
	"fmt"
)

func bubble_sort(A []int, n int) {
	for k := 0; k < n; k++ {
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i]
				swap++
			}
		}
	}
}

var swap int

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	bubble_sort(A, n)
	fmt.Println(swap)
}
```



[Selection Sort](https://www.hackerearth.com/practice/algorithms/sorting/selection-sort/tutorial/)

```go
package main

import (
	"fmt"
)

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

var n, x int

func main() {
	fmt.Scanf("%d %d", &n, &x)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	selection_sort(A, n)
	for i := 0; i < n; i++ {
		fmt.Printf("%d ", A[i])
	}
}

```

