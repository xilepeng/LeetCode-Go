
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














## TEST YOUR UNDERSTANDING


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

func heap_sort(Arr []int, heap_size int) {
	// heap_size := len(Arr)
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

func main() {
	n, count := 0, 0
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
		count++

		if count < 3 {
			fmt.Printf("-1\n")
		} else {
			heap_sort(A, count)
			j := count - 1
			for i := 1; i <= 3; i++ {
				fmt.Printf("%d ", A[j])
				j--
			}
			fmt.Printf("\n")
		}
	}
}


```



[Merge Sort](https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/tutorial/)

## 注意：fmt.Scanf 将导致 Time limit exceeded

### 方法一：Merge Sort

```go
package main

import (
	"fmt"
	"os"
	"io/ioutil"
)

func merge_sort(A, temp []int, start, end int) int {
    if start >= end {
        return 0
    }
    mid := start + (end - start)>>1
    count := merge_sort(A, temp, start, mid) + merge_sort(A, temp, mid+1, end)
    i, j, k := start, mid+1, 0
    for ; i <= mid && j <= end; k++ {
        if A[i] > A[j] {
            count += mid-i+1
            temp[k] = A[j]
            j++
        } else {
            temp[k] = A[i]
            i++
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
    return count
}

func main() {
	// var n int
	// fmt.Scanf("%d", &n)
	// A, temp := make([]int, n), make([]int, n)
	// for i := 0; i < n; i++ {
	// 	fmt.Scanf("%d", &A[i]) //fmt.Scanf 将导致 Time limit exceeded
	// }
	var n int
	_, _ = fmt.Scanln(&n)
	b, _ := ioutil.ReadAll(os.Stdin)
	A, temp := make([]int, n), make([]int, n)
	num, i := 0, 0
	for _, by := range b {
		if by == ' ' {
			A[i] = num
			num = 0
			i++
		} else {
			num = num*10 + int(by-'0')
		}
	}
	A[i] = num

	fmt.Println(merge_sort(A, temp, 0, len(A)-1))
}

```

### 方法二：Merge Sort

```go
package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func merge_sort(A []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end - start) >> 1
	left := merge_sort(A, start, mid)
	right := merge_sort(A, mid+1, end)
	cross := merge(A, start, mid, end)
	return left + right + cross
}
func merge(A []int, start, mid, end int) int {
	temp, count := []int{}, 0
	i, j := start, mid+1
	for i <= mid && j <= end {
		if A[i] > A[j] {
			count += mid - i + 1
			temp = append(temp, A[j])
			j++
		} else {
			temp = append(temp, A[i])
			i++
		}
	}
	for ; i <= mid; i++ {
		temp = append(temp, A[i])
	}
	for ; j <= end; j++ {
		temp = append(temp, A[j])
	}
	copy(A[start:end+1], temp)
	return count
}

func main() {
	var n int
	_, _ = fmt.Scanln(&n)
	b, _ := ioutil.ReadAll(os.Stdin)
	A := make([]int, n) //fmt.Scanf 将导致 Time limit exceeded
	num, i := 0, 0
	for _, by := range b {
		if by == ' ' {
			A[i] = num
			num = 0
			i++
		} else {
			num = num*10 + int(by-'0')
		}
	}
	A[i] = num

	fmt.Println(merge_sort(A, 0, n-1))
}

```

### 方法三：

```go
package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func merge_sort(A, Arr []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)>>1
	left := merge_sort(A, Arr, start, mid)
	right := merge_sort(A, Arr, mid+1, end)
	cross := merge(A, Arr, start, mid, end)
	return left + right + cross
}
func merge(A, Arr []int, start, mid, end int) int {
	p, q, k, count := start, mid+1, 0, 0
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
			count += mid - p + 1
		}
		k++
	}
	for p = 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
	return count
}

func main() {
	var n int
	_, _ = fmt.Scanln(&n)
	b, _ := ioutil.ReadAll(os.Stdin)
	A, Arr := make([]int, n), make([]int, n)
	num, i := 0, 0
	for _, by := range b {
		if by == ' ' {
			A[i] = num
			num = 0
			i++
		} else {
			num = num*10 + int(by-'0')
		}
	}
	A[i] = num

	fmt.Println(merge_sort(A, Arr, 0, n-1))
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
				count++
			}
		}
	}
}

var count int

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	bubble_sort(A, n)
	fmt.Println(count)
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


[Insertion Sort](https://www.hackerearth.com/practice/algorithms/sorting/insertion-sort/tutorial/#c252800)


```go
package main

import (
	"fmt"
)

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

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A, B, hash := make([]int, n), make([]int, n), make(map[int]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
		B[i] = A[i]
	}

	insertion_sort(A, n)
	for i := 0; i < n; i++ {
		hash[A[i]] = i + 1
	}
	for i := 0; i < n; i++ {
		fmt.Printf("%d ", hash[B[i]])
	}
}

// input
// 5
// 9 7 8 12 10

// 7 8 9 10 12 -- sorted array
// output-- 3 1 2 5 4

```

[Counting Sort](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/)

