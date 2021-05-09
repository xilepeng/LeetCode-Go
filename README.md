## Sorting

### 1. Quick Sort

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
	piv, i := A[start], start+1			//第一个元素作为枢轴
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {					//小于枢轴的放一边、大于枢轴的放另一边
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



### 2. Heap Sort

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
	if i != largest { 						//找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] //堆顶调整为最大值
		max_heapify(A, largest, heap_size)  //递归调整子树
	}
}
```


### 3. Merge Sort

合并排序是一种分而治之的算法，其思想是将一个列表分解为几个子列表，直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。

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
		if p > mid { 				//检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { 		//检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { 	//检查哪一部分有更小的元素
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
		for j > 0 && temp < A[j-1] { //左边元素大于当前元素
			A[j] = A[j-1] 			 //向前移动左边元素->
			j--
		}
		A[j] = temp 				 //移动当前元素到正确的位置
	}
}
```

### 5. Bubble Sort

最大元素冒泡到最后

```go
func bubble_sort(A []int, n int) {
	for k := 0; k < n-1; k++ { 		 //按对反复比较调整元素交换它们位置在无序区
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i]
			}
		}
	}
}
```


### 6. Selection Sort

在无序数组中找最小或最大元素并将它们放在有序数组正确的位置

```go
func selection_sort(A []int, n int) {
	for i := 0; i < n-1; i++ {
		min := i                     //假设第一个元素是最小元素
		for j := i + 1; j < n; j++ { //给定无序数组的有效尺寸
			if A[j] < A[min] { 		 //寻找最小元素
				min = j
			}
		}
		A[i], A[min] = A[min], A[i]  //放置最小元素到恰当位置
	}
}
```
