
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
func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heap_size--
		heapify(A, 0, heap_size)
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- {
		heapify(A, i, heap_size)
	}
}
func heapify(A []int, i, heap_size int) {
	lson, rson, largest := i<<1+1, i<<1+2, i
	for lson < heap_size && A[largest] < A[lson] {
		largest = lson
	}
	for rson < heap_size && A[largest] < A[rson] {
		largest = rson
	}
	if i != largest {
		A[i], A[largest] = A[largest], A[i]
		heapify(A, largest, heap_size)
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




![常见的排序算法的时间复杂度.png](http://ww1.sinaimg.cn/large/007daNw2ly1go45x0kga6j32a017e4cd.jpg)











------

### 排序高频题

[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[785. 快速排序](https://www.acwing.com/problem/content/description/787/)

[Quick Sort](https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/tutorial/)


[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)
------



[补充题4. 手撕快速排序 912. 排序数组 ](https://leetcode-cn.com/problems/sort-an-array/)

* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

### 方法一：快速排序

思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

时间复杂度： O(nlog(n)) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := partition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func partition(a []int, l, r int) int {
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] < x {
			i++
			a[i], a[j] = a[j], a[i] //逆序 交换
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```

```go
func sortArray(nums []int) []int {
	rand.Seed(time.Now().UnixNano())
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int()%(r-l+1) + l
	a[i], a[r] = a[r], a[i]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] < x {
			i++
			a[i], a[j] = a[j], a[i] //逆序 交换
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```


思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

### 方法一：快速排序

```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := partition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func partition(a []int, l, r int) int {
	a[r], a[(l+r)>>1] = a[(l+r)>>1], a[r]
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] <= x { //逆序交换
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```

- 时间复杂度： O(nlog(n)) 
- 空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

### 方法二：快速排序
```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		a[(l+r)>>1], a[r] = a[r], a[(l+r)>>1]
		i := l - 1
		for j := l; j < r; j++ {
			if a[j] <= a[r] { //逆序交换
				i++
				a[i], a[j] = a[j], a[i]
			}
		}
		i++
		a[i], a[r] = a[r], a[i]
		quickSort(a, l, i-1)
		quickSort(a, i+1, r)
	}
}
```

### 方法三：基于随机选取主元的快速排序

```go
func sortArray(nums []int) []int {
	rand.Seed(time.Now().UnixNano())
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int()%(r-l+1) + l
	a[r], a[i] = a[i], a[r]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] <= x { //逆序交换
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}

```

复杂度分析

- 时间复杂度：基于随机选取主元的快速排序时间复杂度为期望 O(nlogn)，其中 n 为数组的长度。详细证明过程可以见《算法导论》第七章，这里不再大篇幅赘述。

- 空间复杂度：O(h)，其中 h 为快速排序递归调用的层数。我们需要额外的 O(h) 的递归调用的栈空间，由于划分的结果不同导致了快速排序递归调用的层数也会不同，最坏情况下需 O(n) 的空间，最优情况下每次都平衡，此时整个递归树高度为 logn，空间复杂度为 O(logn)。


[785. 快速排序](https://www.acwing.com/problem/content/description/787/)
```go
package main 

import "fmt"

func quickSort(q []int, l, r int) {
    if l >= r { // 终止条件
        return
    }
    x := q[(l+r)>>1] // 确定分界点
    i, j := l-1, r+1  // 两个指针，因为do while要先自增/自减
    for i < j {       // 每次迭代
        for { // do while 语法
            i++ // 交换后指针要移动，避免没必要的交换
            if q[i] >= x {
                break
            }
        }
        for {
            j--
            if q[j] <= x {
                break
            }
        }
        if i < j { // swap 两个元素
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j) // 递归处理左右两段
    quickSort(q, j+1, r)
}

func main() {
    var n int 
    fmt.Scanf("%d", &n)
    q := make([]int, n)
    for i := 0; i < n; i ++ {
        fmt.Scanf("%d", &q[i])
    }
    quickSort(q, 0, n-1)
    for i := 0; i < n; i ++ {
        fmt.Printf("%d ",q[i] )
    }
}


```


[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)


* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

## 方法一：基于快速排序的选择方法

快速选择算法思路：

1. 随机确定分界点 p
2. 调整区间：pIdx <= index, 递归右子区间; pIdx <= index, 递归右边
3. 递归处理左边或右边 
只要某次划分的 q 为倒数第 k 个下标的时候，我们就已经找到了答案。
如果划分得到的 q 正好就是我们需要的下标，就直接返回 a[q]；
否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。

时间复杂度： O(n) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

```go
func findKthLargest(nums []int, k int) int {
    rand.Seed(time.Now().UnixNano())
	return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}
func quickSelect(a []int, l, r, index int) int {
	p := randomPartition(a, l, r)
	if p == index {
		return a[p]
	} else if p < index {
		return quickSelect(a, p+1, r, index)
	} 
	return quickSelect(a, l, p-1, index)
}

func randomPartition(a[]int, l, r int) int {
    i := rand.Int() % (r - l + 1) + l 
    a[r], a[i] = a[i], a[r]
    return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	i := l - 1
	for j := l; j < r; j++ {
		if a[j] <= a[r] {
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```


## 方法二：基于堆排序的选择方法

思路和算法

建立一个大根堆，做 k - 1 次删除操作后堆顶元素就是我们要找的答案。

```go
func findKthLargest(nums []int, k int) int {
    heapSize := len(nums)
    buildMaxHeap(nums, heapSize)
    for i := len(nums) - 1; i >= len(nums) - k + 1; i-- {
        nums[0], nums[i] = nums[i], nums[0]
        heapSize--
        maxHeapify(nums, 0, heapSize)
    }
    return nums[0]
}

func buildMaxHeap(a []int, heapSize int) {
    for i := heapSize/2; i >= 0; i-- {
        maxHeapify(a, i, heapSize)
    }
}

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i
    if l < heapSize && a[l] > a[largest] {
        largest = l
    }
    if r < heapSize && a[r] > a[largest] {
        largest = r
    }
    if largest != i {
        a[i], a[largest] = a[largest], a[i]
        maxHeapify(a, largest, heapSize)
    }
}

```

复杂度分析

- 时间复杂度：O(nlogn)，建堆的时间代价是 O(n)，删除的总代价是 O(klogn)，因为 k < n，故渐进时间复杂为 O(n+klogn)=O(nlogn)。
- 空间复杂度：O(logn)，即递归使用栈空间的空间代价。




[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)


### 方法一：堆排序（大根堆）
思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。


```go
func sortArray(nums []int) []int {
	heapSort(nums)
	return nums
}
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
		maxHeapify(a, largest, heapSize)    //递归调整子树
	}
}
```

复杂度分析

- 时间复杂度：O(nlogn)。初始化建堆的时间复杂度为 O(n)，建完堆以后需要进行 n−1 次调整，一次调整（即 maxHeapify） 的时间复杂度为 O(logn)，那么 n−1 次调整即需要 O(nlogn) 的时间复杂度。因此，总时间复杂度为 O(n+nlogn)=O(nlogn)。

- 空间复杂度：O(1)。只需要常数的空间存放若干变量。






[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)



### 方法一：归并排序

思路

归并排序利用了分治的思想来对序列进行排序。
对一个长为 n 的待排序的序列，我们将其分解成两个长度为 n/2 的子序列。
每次先递归调用函数使两个子序列有序，然后我们再线性合并两个有序的子序列使整个序列有序。

1. 确定分解点: mid := (l + r) / 2
2. 递归排序左右两边
3. 归并


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

### 方法二：归并排序
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
	mergeSort(nums, 0, n-1)
	return nums
}
func mergeSort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		mergeSort(A, start, mid)
		mergeSort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	temp := []int{}
	i, j := start, mid+1
	for k := start; k <= end; k++ {
		if i > mid {
			temp = append(temp, A[j])
			j++
		} else if j > end {
			temp = append(temp, A[i])
			i++
		} else if A[i] <= A[j] {
			temp = append(temp, A[i])
			i++
		} else {
			temp = append(temp, A[j])
			j++
		}
	}
	copy(A[start:end+1], temp)
}
```


### 方法二：归并排序

```go
func sortArray(nums []int) []int {
	mergeSort(nums, 0, len(nums)-1)
	return nums
}
func mergeSort(a []int, l, r int) {
	if l < r {
		mid := (l + r) >> 1
		mergeSort(a, l, mid)
		mergeSort(a, mid+1, r)
		merge(a, l, mid, r)
	}
}
func merge(a []int, l, mid, r int) {
	tmp := []int{}
	i, j := l, mid+1
	for i <= mid || j <= r {
		if i > mid || j <= r && a[j] < a[i] {
			tmp = append(tmp, a[j])
			j++
		} else {
			tmp = append(tmp, a[i])
			i++
		}
	}
	copy(a[l:r+1], tmp)
}
```

```go
func sortArray(nums []int) []int {
	mergeSort(nums, 0, len(nums)-1)
	return nums
}
func mergeSort(a []int, l, r int) {
	if l < r {
		mid := (l + r) >> 1
		mergeSort(a, l, mid)
		mergeSort(a, mid+1, r)
		tmp := []int{}
		i, j := l, mid+1
		for i <= mid || j <= r {
			if i > mid || j <= r && a[j] < a[i] {
				tmp = append(tmp, a[j])
				j++
			} else {
				tmp = append(tmp, a[i])
				i++
			}
		}
		copy(a[l:r+1], tmp)
	}
}
```


复杂度分析

- 时间复杂度：O(nlogn)。由于归并排序每次都将当前待排序的序列折半成两个子序列递归调用，然后再合并两个有序的子序列，而每次合并两个有序的子序列需要 O(n) 的时间复杂度，所以我们可以列出归并排序运行时间 T(n) 的递归表达式：

T(n)=2T(n/2)+O(n)

​ 根据主定理我们可以得出归并排序的时间复杂度为 O(nlogn)。

- 空间复杂度：O(n)。我们需要额外 O(n) 空间的 tmp 数组，且归并排序递归调用的层数最深为 log_2 n，所以我们还需要额外的 O(logn) 的栈空间，所需的空间复杂度即为 O(n+logn)=O(n)。
















------



![常见的排序算法的时间复杂度.png](http://ww1.sinaimg.cn/large/007daNw2ly1go45x0kga6j32a017e4cd.jpg)

1. 快速排序

```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := partition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func partition(a []int, l, r int) int {
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] < x {
			i++
			a[i], a[j] = a[j], a[i] //逆序 交换
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```

```go
func sortArray(nums []int) []int {
    rand.Seed(time.Now().UnixNano())
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int() % (r - l + 1) + l 
	a[i], a[r] = a[r], a[i]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l - 1
	for j := l; j < r; j ++ {
		if a[j] <= x {
			i ++ 
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i+1
}

```



```go
func sortArray(nums []int) []int {
    quickSort(nums, 0, len(nums)-1)
    return nums
}
func quickSort (nums[]int, l, r int) {
    if l >= r {
        return
    }
    p := nums[(l+r)>>1]
    i, j := l-1, r+1
    for i < j {
        for {
            i ++ 
            if nums[i] >= p {
                break
            }
        }
        for {
            j --
            if nums[j] <= p {
                break
            }
        }
        if i < j {
            nums[i], nums[j] = nums[j], nums[i]
        }
    }
    quickSort(nums, l, j)
    quickSort(nums, j+1, r)
}
```


2. 堆排序

```go
func heapSort(a []int) {
    heapSize := len(a) 
    buildMaxHeap(a, heapSize)
    for i := heapSize - 1; i >= 1; i -- {
        a[0], a[i] = a[i], a[0]
        heapSize -- 
        maxHeapify(a, 0, heapSize)
    }
}

func buildMaxHeap(a[]int, heapSize int) {
    for i := heapSize / 2; i >= 0; i -- {
        maxHeapify(a, i, heapSize)
    }
} 

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i 
    if l < heapSize && a[largest] < a[l] {
        largest = l 
    } 
    if r < heapSize && a[largest] < a[r] {
        largest = r 
    }
    if largest != i {
        a[largest], a[i] = a[i], a[largest]
        maxHeapify(a, largest, heapSize)
    }
}
```

3. 归并排序

```go
func mergeSort(a []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) / 2 
	mergeSort(a, l, mid)
	mergeSort(a, mid + 1, r)
    tmp := []int {}
	i, j := l, mid + 1
	for i <= mid || j <= r {
		if i > mid || (j <= r && a[j] < a[i]) {
			tmp = append(tmp, a[j])
			j++
		} else {
			tmp = append(tmp, a[i])
			i++
		}
	}
	copy(a[l: r + 1], tmp)
}
```

4. 选择排序

```go
func selectSort(a []int) {
	for i := 0; i < len(a)-1; i++ {
		minIdx := i
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[minIdx] {
				minIdx = j
			}
		}
		a[i], a[minIdx] = a[minIdx], a[i]
	}
}
```

5. 插入排序

```go
func insertionSort(nums []int) {
	for i := 1; i < len(nums); i++ {
		tmp := nums[i]
		j := i - 1
		for j >= 0 && nums[j] > tmp {
			nums[j+1] = nums[j] //向后移动1位
			j--                 //向前扫描
		}
		nums[j+1] = tmp //添加到小于它的数的右边
	}
}
```

6. 冒泡排序

```Golang
func bubble_sort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ { //最后剩一个数不需比较-1
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}
```

7. 计数排序

```Golang
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

8. 桶排序

```go

```

9. 基数排序

```go

```





















------


```go
func sortArray(nums []int) []int {
    heapSort(nums)
    return nums
}

func heapSort(a []int) {
    heapSize := len(a) - 1
    buildMaxHeap(a, heapSize)
    for i := heapSize; i >= 1; i -- {
        a[0], a[i] = a[i], a[0]
        heapSize -- 
        maxHeapify(a, 0, heapSize)
    }
}

func buildMaxHeap(a[]int, heapSize int) {
    for i := heapSize / 2; i >= 0; i -- {
        maxHeapify(a, i, heapSize)
    }
} 

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i 
    if l <= heapSize && a[largest] < a[l] {
        largest = l 
    } 
    if r <= heapSize && a[largest] < a[r] {
        largest = r 
    }
    if largest != i {
        a[largest], a[i] = a[i], a[largest]
        maxHeapify(a, largest, heapSize)
    }
}
```

```Golang
func quick_sort(nums []int, l, r int) {
	if l >= r {
		return
	}
	nums[r], nums[(l+r)>>1] = nums[(l+r)>>1], nums[r]
	i := l - 1
	for j := l; j < r; j++ {
		if nums[j] < nums[r] {
			i++
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
	i++
	nums[i], nums[r] = nums[r], nums[i]
	quick_sort(nums, l, i-1)
	quick_sort(nums, i+1, r)
}
```

```Golang
func merge_sort(nums []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) >> 1
	merge_sort(nums, l, mid)
	merge_sort(nums, mid+1, r)
	i, j := l, mid+1
	tmp := []int{}
	for i <= mid || j <= r {
		if i > mid || (j <= r && nums[j] < nums[i]) {
			tmp = append(tmp, nums[j])
			j++
		} else {
			tmp = append(tmp, nums[i])
			i++
		}
	}
	copy(nums[l:r+1], tmp)
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

```Golang
func heap_sort(nums []int) {
	lens := len(nums) - 1
	for i := lens << 1; i >= 0; i-- {//建堆O(n)
		down(nums, i, lens)
	}
	for j := lens; j >= 1; j-- {
		nums[0], nums[j] = nums[j], nums[0]
		lens--
		down(nums, 0, lens)
	}
}
func down(nums []int, i, lens int) {//O(logn)
	max := i
	if i<<1+1 <= lens && nums[i<<1+1] > nums[max] {
		max = i<<1 + 1
	}
	if i<<1+2 <= lens && nums[i<<1+2] > nums[max] {
		max = i<<1 + 2
	}
	if i != max {
		nums[i], nums[max] = nums[max], nums[i]
		down(nums, max, lens)
	}
}
```

```Golang
func select_sort(nums []int) {
	for i := 0; i < len(nums)-1; i++ {
		pos := i
		for j := i + 1; j < len(nums); j++ {
			if nums[j] < nums[pos] {
				pos = j
			}
		}
		nums[i], nums[pos] = nums[pos], nums[i]
	}
}
```

```Golang
func insert_sort(nums []int) {
	for i := 1; i < len(nums); i++ {
		tmp := nums[i]
		j := i - 1
		for j >= 0 && nums[j] > tmp {
			nums[j+1] = nums[j] //向后移动1位
			j--                 //向前扫描
		}
		nums[j+1] = tmp //添加到小于它的数的右边
	}
}
```


```Golang
func bubble_sort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ { //最后剩一个数不需比较-1
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}
```


```Golang
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



------

[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

## 方法一：快速排序

思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

时间复杂度： O(nlog(n)) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。


```go
func sortArray(nums []int) []int {
    rand.Seed(time.Now().UnixNano())
    quickSort(nums, 0, len(nums)-1)
    return nums
}
func quickSort(a []int, l, r int) {
    if l >= r {
        return
    }
    pos := partition(a, l, r)
    quickSort(a, l, pos - 1)
    quickSort(a, pos + 1, r)
}
func partition(a []int, l, r int) int {
    p := rand.Int()%(r - l + 1) + l 
    a[r],a[p] = a[p], a[r] 
    i := l - 1
    for j := l; j < r; j ++ {
        if a[j] <= a[r] {
            i ++ 
            a[i], a[j] = a[j], a[i]
        }
    }
    a[i+1], a[r] = a[r], a[i+1]
    return i + 1
}
```

## 方法二：堆排序

思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，
我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。



```go
func heapSort(a []int) {
    heapSize := len(a) 
    buildMaxHeap(a, heapSize)
    for i := heapSize - 1; i >= 1; i -- {
        a[0], a[i] = a[i], a[0]
        heapSize -- 
        maxHeapify(a, 0, heapSize)
    }
}

func buildMaxHeap(a[]int, heapSize int) {
    for i := heapSize / 2; i >= 0; i -- {
        maxHeapify(a, i, heapSize)
    }
} 

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i 
    if l < heapSize && a[largest] < a[l] {
        largest = l 
    } 
    if r < heapSize && a[largest] < a[r] {
        largest = r 
    }
    if largest != i {
        a[largest], a[i] = a[i], a[largest]
        maxHeapify(a, largest, heapSize)
    }
}
```



## 方法三：归并排序
思路

归并排序利用了分治的思想来对序列进行排序。
对一个长为 n 的待排序的序列，我们将其分解成两个长度为 n/2 的子序列。
每次先递归调用函数使两个子序列有序，然后我们再线性合并两个有序的子序列使整个序列有序。

1. 确定分解点: mid := (l + r) / 2
2. 递归排序左右两边
3. 归并

```go
func sortArray(nums []int) []int {
	mergeSort(nums, 0, len(nums)-1)
	return nums
}
func mergeSort(a []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) / 2
	mergeSort(a, l, mid)
	mergeSort(a, mid+1, r)
	tmp := []int{}
	i, j := l, mid+1
	for i <= mid || j <= r {
		if i > mid || (j <= r && a[j] < a[i]) {
			tmp = append(tmp, a[j])
			j++
		} else {
			tmp = append(tmp, a[i])
			i++
		}
	}
	copy(a[l:r+1], tmp)
}
```

## 方法4：选择排序

从未排序序列中找到最小元素，存放到排序序列的起始位置
再从剩余元素中找到最小元素，存放到已排序序列末尾...


```go
func sortArray(nums []int) []int {
	selectSort(nums)
	return nums
}

func selectSort(a []int) {
	for i := 0; i < len(a)-1; i++ {
		minIdx := i
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[minIdx] {
				minIdx = j
			}
		}
		a[i], a[minIdx] = a[minIdx], a[i]
	}
}
```
Time Limit Exceeded
11/11 cases passed (N/A)

# 5.插入排序

每次将一个待排序的序列插入到一个前面已排好序的子序列当中

构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
选取数组第2个元素开始比较，如果左边第1个元素比它大，左边元素向右移动1位，索引减1，向前扫描...
直到左边元素比他小，插入这个元素右边

```go
func insertionSort(nums []int) {
	for i := 1; i < len(nums); i++ {
		tmp := nums[i]
		j := i - 1
		for j >= 0 && nums[j] > tmp {
			nums[j+1] = nums[j] //向后移动1位
			j--                 //向前扫描
		}
		nums[j+1] = tmp //添加到小于它的数的右边
	}
}
```














------



# 1. 快速排序

### 模板一：

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

时间复杂度： O(nlog(n)) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

C++ 模板一：

```c++ []
void quickSort(int a[], int l, int r) {
	if (l >= r) return;
	int i = l-1, j =r+1, x = a[l+r>>1];
	while(i < j) {
		do i++; while a[i] < x;
		do j--; while a[j] > x;
		if(i<j) swap(a[i],a[j]);
	}
	quickSort(a, l, j);
	quickSort(a, j+1, r);
}
```
C++ 模板二：

```c++ []
void quickSort(int a[], int l, int r) {
	if (l >= r) return;
	int i = l-1, j =r+1, x = a[l+r>>1];
	while(i < j) {
		while a[++ i] < x;
		while a[-- j] > x;
		if(i<j) swap(a[i],a[j]);
	}
	quickSort(a, l, j);
	quickSort(a, j+1, r);
}
```

go 模板一：

```go []
func sortArray(nums []int) []int {
    quickSort(nums, 0, len(nums)-1)
    return nums
}
func quickSort (q[]int, l, r int) {
    if l >= r {
        return
    }
    p := q[(l+r)>>1]
    i, j := l-1, r+1
    for i < j {
        for {
            i ++ 
            if q[i] >= p {
                break
            }
        }
        for {
            j -- 
            if q[j] <= p {
                break
            }
        }
        if i < j {
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j)
    quickSort(q, j+1, r)
}
```

go 模板二：

```go []
func sortArray(nums []int) []int {
    quickSort(nums, 0, len(nums)-1)
    return nums
}
func quickSort (q[]int, l, r int) {
    if l >= r {
        return
    }
    p := q[(l+r)>>1]
    i, j := l-1, r+1
    for i < j {
        for {
            i ++ 
            if q[i] >= p {
                break
            }
        }
        for {
            j -- 
            if q[j] <= p {
                break
            }
        }
        if i < j {
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j)
    quickSort(q, j+1, r)
}
```
时间复杂度： O(n)
n + n/2 + n/4 + ... = (1 + 1/2 + 1/4 + ...)n <= 2n 

## 方法二

```go
func sortArray(nums []int) []int {
    rand.Seed(time.Now().UnixNano())
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int() % (r - l + 1) + l 
	a[i], a[r] = a[r], a[i]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l - 1
	for j := l; j < r; j ++ {
		if a[j] <= x {
			i ++ 
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i+1
}
```

## 方法三： 


```go []
func sortArray(nums []int) []int {
    quickSort(nums, 0, len(nums)-1)
    return nums
}
func quickSort(a []int, l, r int) {
    if l >= r {
        return
    }
    a[r], a[(l+r)>>1] = a[(l+r)>>1], a[r]
    i := l - 1
    for j := l; j < r; j ++ {
        if a[j] < a[r] {
            i ++
            a[i], a[j] = a[j], a[i]
        }
    }
    i ++
    a[i], a[r] = a[r], a[i]
    quickSort(a, l, i-1 )
    quickSort(a, i+1, r)
}
```

```go []
func sortArray(nums []int) []int {
    quickSort(nums, 0, len(nums)-1)
    return nums
}
func quickSort(nums []int, l, r int) {
    if l >= r {
        return
    }
    nums[r], nums[(l+r)>>1] = nums[(l+r)>>1], nums[r]
    i := l - 1
    for j := l; j < r; j++ {
        if nums[j] < nums[r] {
            i++
            nums[i], nums[j] = nums[j], nums[i]
        }
    }
    i ++
    nums[i], nums[r] = nums[r], nums[i]
    quickSort(nums, l, i-1)
    quickSort(nums, i+1, r)
}
```



```go
func quick_sort(A []int, start, end int) {
	if start >= end {
		return
	}
	piv := A[start+(end-start)>>1]
	i, j := start-1, end+1
	for i < j {
		for {
			i++
			if A[i] >= piv {
				break
			}
		}
		for {
			j--
			if A[j] <= piv {
				break
			}
		}
		if i < j {
			A[i], A[j] = A[j], A[i]
		}
	}
	quick_sort(A, start, j)
	quick_sort(A, j+1, end)
}
```