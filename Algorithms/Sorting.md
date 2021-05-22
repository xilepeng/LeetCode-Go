
## Sorting

### 1. [Quick Sort](https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/tutorial/)


快速排序基于分而治之的方法，随机选择枢轴元素划分数组，左边小于枢轴、右边大于枢轴，递归处理左右两边

```go
func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}
func partition(A []int, start, end int) int {
	piv, i := A[start], start+1//第一个元素作为枢轴
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {//小于枢轴的放一边、大于枢轴的放另一边
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[start], A[i-1] = A[i-1], A[start] //放置枢轴到正确的位置
	return i - 1 						//返回枢轴的位置
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().Unix())
	random := start + rand.Int()%(end-start+1)
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```




### 2. [Heap Sort](https://www.hackerearth.com/practice/algorithms/sorting/heap-sort/tutorial/)

在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序

```go
func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]      //交换堆顶与堆底元素，最大值放置在数组末尾
		heap_size--                  //剩余待排序元素整理成堆
		max_heapify(A, 0, heap_size) //堆顶 root 向下调整
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- { // heap_size / 2 后面都是叶子节点，不需要向下调整
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) {
	lson, rson, largest := i<<1+1, i<<1+2, i
	for lson < heap_size && A[largest] < A[lson] { //左儿子存在并大于根
		largest = lson
	}
	for rson < heap_size && A[largest] < A[rson] { //右儿子存在并大于根
		largest = rson
	}
	if i != largest { //找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] //堆顶调整为最大值
		max_heapify(A, largest, heap_size)  //递归调整子树
	}
}
```



### 3. [Merge Sort](https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/tutorial/)

归并排序是一种分而治之的算法，其思想是将一个列表分解为几个子列表，直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。

```go
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1 //分2部分定义当前数组
		merge_sort(A, start, mid)     //排序数组的第1部分
		merge_sort(A, mid+1, end)     //排序数组的第2部分
		merge(A, start, mid, end)     //通过比较2个部分的元素来合并2个部分
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid { //检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { //检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { //检查哪一部分有更小的元素
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
func merge_sort(A []int, start, end int) {
	if start >= end {
		return
	}
	mid := start + (end-start)>>1
	merge_sort(A, start, mid)
	merge_sort(A, mid+1, end)

	Arr := make([]int, end-start+1)
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
	copy(A[start:end+1], Arr)
}
```



### 4. [Insertion Sort](https://www.hackerearth.com/practice/algorithms/sorting/insertion-sort/tutorial/#c252800)

插入排序基于这样的想法：每次迭代都会消耗输入元素中的一个元素，以找到其正确位置，即该元素在排序数组中的位置。

通过在每次迭代时增加排序后的数组来迭代输入元素。它将当前元素与已排序数组中的最大值进行比较。如果当前元素更大，则它将元素留在其位置，然后移至下一个元素，否则它将在已排序数组中找到其正确位置，并将其移至该位置。这是通过将已排序数组中所有大于当前元素的元素移动到前面的一个位置来完成的

```go
func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] { //当前元素小于左边元素
			A[j] = A[j-1] //向前移动左边元素->
			j--
		}
		A[j] = temp //移动当前元素到正确的位置
	}
}
```





### 5. [Bubble Sort](https://www.hackerearth.com/practice/algorithms/sorting/bubble-sort/tutorial/)


反复比较成对的相邻元素，交换它们的位置如果他们在无序区。（最大元素冒泡到最后）

```go
func bubble_sort(A []int, n int) {
	for k := 0; k < n-1; k++ {  // (n-k-1) 是忽略比较的元素，这些元素已比较完成在简单的迭代中
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i] //交换
			}
		}
	}
}
```


### 6. [Selection Sort](https://www.hackerearth.com/practice/algorithms/sorting/selection-sort/tutorial/)

在未排序的数组中找到最小或最大元素，然后将其放在已排序的数组中的正确位置。

```go
func selection_sort(A []int, n int) {
	for i := 0; i < n-1; i++ {		 //在每次迭代中将数组的有效大小减少1
		min := i                     //假设第一个元素是未排序数组的最小值
		for j := i + 1; j < n; j++ { //给出未排序数组的有效大小
			if A[j] < A[min] { //找到最小的元素
				min = j
			}
		}
		A[i], A[min] = A[min], A[i] //将最小元素放在适当的位置
	}
}
```



### 7. [Counting Sort](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/)

```go
func counting_sort(A, Aux, sortedA []int, N int) {
	K := 0
	for i := 0; i < N; i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// for i := 0; i <= K; i++ {
	// 	Aux[i] = 0
	// }
	for i := 0; i < N; i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			sortedA[j] = i
			j++
		}
	}
}
```


```go
func count_sort(nums []int) {
	cnt := [100001]int{}
	for i := 0; i < len(nums); i++ {
		cnt[nums[i]+50000] ++ //防止负数导致数组越界
	}
	for i, idx := 0, 0; i < 100001; i++ {
		for cnt[i] > 0 {
			nums[idx] = i - 50000
			idx++
			cnt[i] --
		}
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
	rand.Seed(time.Now().Unix())
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
	lson, rson, largest := i<<1+1, i<<1+2, i
	for lson < heap_size && A[largest] < A[lson] {
		largest = lson
	}
	for rson < heap_size && A[largest] < A[rson] {
		largest = rson
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


```go
package main

import (
	"fmt"
	"math"
)

func counting_sort(A, Aux, sortedA []int, N int) {
	K := 0
	for i := 0; i < N; i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// for i := 0; i <= K; i++ {
	// 	Aux[i] = 0
	// }
	for i := 0; i < N; i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			sortedA[j] = i
			j++
		}
	}
}

func main() {
	var n int

	fmt.Scanf("%d", &n)
	A, sortedA := make([]int, n), make([]int, n)
	max := math.MinInt64
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
		if A[i] > max {
			max = A[i]
		}
	}
	Aux := make([]int, max+1)
	counting_sort(A, Aux, sortedA, n)
	for i := 0; i < n; i++ {
		tmp := Aux[sortedA[i]]
		if tmp > 0 {
			fmt.Printf("%d %d\n", sortedA[i], Aux[sortedA[i]])
		}
		for tmp -= 1; tmp > 0; i++ {
			tmp--
		}
	}
}

```

